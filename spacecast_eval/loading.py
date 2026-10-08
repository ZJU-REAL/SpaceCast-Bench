"""HF/local Parquet loading, leaving image decoding until inference."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

from .data import normalize, read_rows, source_type, validate_view_metadata


def add_data_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Hugging Face dataset repository")
    group.add_argument("--parquet", help="Parquet file or dataset directory")
    group.add_argument("--jsonl", help="Source annotation JSONL (requires --image-root for inference)")
    parser.add_argument("--config", default=None, help="HF dataset configuration")
    parser.add_argument("--revision", default=None, help="HF dataset commit or revision")
    parser.add_argument("--split", default="test")


def load_benchmark(*, dataset=None, parquet=None, jsonl=None, config=None, revision=None, split="test"):
    from datasets import Dataset, Image, List, load_dataset
    if sum(x is not None for x in (dataset, parquet, jsonl)) != 1:
        raise ValueError("Specify exactly one dataset, parquet, or jsonl source")
    if dataset:
        ds = load_dataset(dataset, name=config, revision=revision, split=split)
    elif parquet:
        path = Path(parquet)
        files = ([path] if path.is_file() else sorted((path / "data").glob(f"{split}-*.parquet")))
        if not files and path.is_dir():
            files = sorted(path.glob("*.parquet"))
        if not files or any(p.suffix != ".parquet" for p in files):
            raise ValueError(f"No {split} Parquet shards found at {path}")
        ds = load_dataset("parquet", data_files={split: [str(p) for p in files]}, split=split)
    else:
        ds = Dataset.from_list([normalize(row) for row in read_rows(jsonl)])
    if "images" in ds.column_names:
        ds = ds.cast_column("images", List(Image(decode=False)))
    required = {"question_uid", "question", "options", "answer", "type", "level", "multi_select"}
    if not required <= set(ds.column_names):
        raise ValueError(f"Missing benchmark columns: {sorted(required - set(ds.column_names))}")
    if not len(ds):
        raise ValueError("Empty benchmark")
    rows = ds.select_columns([c for c in ds.column_names if c != "images"]).to_list()
    seen = set()
    from .protocol import parse_prediction
    for row in rows:
        uid = row["question_uid"]
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError(f"Empty or duplicate question UID: {uid}")
        seen.add(uid)
        source_type(row)
        validate_view_metadata(row)
        if (not isinstance(row["options"], list) or not 2 <= len(row["options"]) <= 26 or
                not all(isinstance(v, str) for v in row["options"])):
            raise ValueError(f"Invalid options: {uid}")
        if parse_prediction(row["answer"], [chr(65+i) for i in range(len(row["options"]))], row["multi_select"]) is None:
            raise ValueError(f"Invalid gold answer: {uid}")
    return ds


def from_args(args):
    return load_benchmark(dataset=args.dataset, parquet=args.parquet, jsonl=args.jsonl,
                          config=args.config, revision=args.revision, split=args.split)


def data_source(args):
    """Where the gold data came from, recorded in reports for provenance."""
    if args.dataset:
        return {"kind": "hf", "dataset": args.dataset, "config": args.config,
                "revision": args.revision, "split": args.split}
    # The path as given, not resolved: reports get shared, and an absolute path
    # would expose the local user and directory layout.
    path = args.parquet or args.jsonl
    return {"kind": "parquet" if args.parquet else "jsonl", "path": str(path), "split": args.split}


def gold_rows(ds):
    return ds.select_columns([c for c in ds.column_names if c != "images"]).to_list()


def fingerprint(ds):
    """Content hash over the gold annotations, used to guard resume.

    Built from public data only. `datasets` has spelled its own dataset
    fingerprint both publicly and privately across releases, so relying on
    `_fingerprint` risked resuming silently against a changed benchmark if that
    attribute moved again. Question text, options, answers and image references
    are what a resume must stay pinned to, and hashing them is stable.
    """
    payload = {"gold": gold_rows(ds)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def resolve_image(root, relative):
    path = PurePosixPath(relative)
    if (path.is_absolute() or "\\" in relative or
            any(p in {"", ".", ".."} for p in relative.split("/"))):
        raise ValueError(f"Unsafe image reference: {relative}")
    base = Path(root).resolve()
    resolved = (base / relative).resolve()
    if base not in resolved.parents:
        raise ValueError(f"Image escapes its root: {relative}")
    return resolved


def image_bytes(row, root=None):
    embedded = row.get("images")
    paths = row.get("image_paths") or []
    validate_view_metadata(row, len(embedded) if embedded is not None else len(paths))
    if embedded is not None:
        if not 1 <= len(embedded) <= 8 or (paths and len(paths) != len(embedded)):
            raise ValueError(f"Incorrect image count: {row['question_uid']}")
        if paths and all(item.get("path") for item in embedded) and [item["path"] for item in embedded] != paths:
            raise ValueError(f"Embedded image order disagrees with references: {row['question_uid']}")
        blobs = []
        for item in embedded:
            if item.get("bytes"):
                blobs.append(item["bytes"])
            elif root and item.get("path"):
                blobs.append(resolve_image(root, item["path"]).read_bytes())
            else:
                raise ValueError("Image bytes are missing; supply an explicit --image-root")
        return blobs
    if not root or not 1 <= len(paths) <= 8:
        raise ValueError("No embedded images; supply --image-root containing dataset/scene/frame.jpg")
    return [resolve_image(root, p).read_bytes() for p in paths]


def decode_images(blobs):
    from PIL import Image
    result = []
    for data in blobs:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG/PNG images are supported")
            image.load()
            result.append(image.convert("RGB"))
    return result
