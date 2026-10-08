"""Export the test annotations as Parquet shards with embedded JPEG/PNG media."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .data import features, normalize, read_rows, sha256, TYPE_NAMES, LEVELS
from .loading import resolve_image
from .scoring import atomic_json


def audit_media(rows, root):
    from PIL import Image
    paths = sorted({p for row in rows for p in row["image_paths"]})
    missing, invalid, hashes = [], [], {}
    for rel in paths:
        path = resolve_image(root, rel)
        if not path.is_file():
            missing.append(rel)
            continue
        try:
            with Image.open(path) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG/PNG supported")
                image.verify()
            hashes[rel] = sha256(path)
        except (OSError, ValueError) as e:
            invalid.append({"path": rel, "error": type(e).__name__})
    return {"questions": len(rows), "unique_images": len(paths), "available": len(hashes),
            "missing": missing, "invalid": invalid, "image_sha256": hashes,
            "complete_questions": sum(all(p in hashes for p in row["image_paths"]) for row in rows)}


def export(source, image_root, output, *, shard_rows=100, expected_questions=3862, audit_path=None):
    from datasets import Dataset, Features, Image, List
    if shard_rows < 1:
        raise ValueError("shard_rows must be positive")
    originals = read_rows(source)
    rows = [normalize(row) for row in originals]
    # source_record_json must reproduce each source record exactly; this is the
    # check behind the manifest's all_source_records_equal flag.
    mismatched = [row["question_uid"] for row, original in zip(rows, originals)
                  if json.loads(row["source_record_json"]) != original]
    if mismatched:
        raise ValueError(f"source_record_json differs from {len(mismatched)} source records, e.g. {mismatched[0]}")
    if len(rows) != expected_questions or len({r["question_uid"] for r in rows}) != len(rows):
        raise ValueError(f"Expected {expected_questions} unique test questions, got {len(rows)}")
    if not rows:
        raise ValueError("Empty benchmark")
    if any(not 1 <= len(row["image_paths"]) <= 8 for row in rows):
        raise ValueError("Each question must have 1 to 8 images")
    report = audit_media(rows, image_root)
    if audit_path:
        atomic_json(audit_path, report)
    if report["missing"] or report["invalid"]:
        raise ValueError(f"Incomplete media: {len(report['missing'])} missing, {len(report['invalid'])} invalid; no questions dropped")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Export requires an empty output directory")
    data_dir = output / "data"
    data_dir.mkdir()
    schema = Features({"images": List(Image()), **features()})
    n = (len(rows)+shard_rows-1)//shard_rows
    files = []
    checked = {"records": 0, "images": 0}
    for index in range(n):
        batch = []
        for row in rows[index*shard_rows:(index+1)*shard_rows]:
            batch.append({"images": [{"bytes": resolve_image(image_root, p).read_bytes(), "path": p}
                                      for p in row["image_paths"]], **row})
        path = data_dir / f"test-{index:05d}-of-{n:05d}.parquet"
        # datasets already passes its own compression default (zstd) to
        # ParquetWriter; supplying it again duplicates the keyword and fails.
        Dataset.from_list(batch, features=schema).to_parquet(str(path), batch_size=100)
        # Validate actual serialized records, byte hashes and viewing order.
        recovered = Dataset.from_parquet(str(path)).cast_column("images", List(Image(decode=False)))
        if len(recovered) != len(batch):
            raise ValueError(f"Parquet shard {path.name} holds {len(recovered)} rows, expected {len(batch)}")
        for before, after in zip(batch, recovered):
            images = after.pop("images")
            if after != {k: v for k, v in before.items() if k != "images"}:
                raise ValueError("Parquet annotation round trip failed")
            if json.loads(after["source_record_json"]) != originals[checked["records"]]:
                raise ValueError(f"Serialized source record differs: {after['question_uid']}")
            if [v["path"] for v in images] != before["image_paths"]:
                raise ValueError("Parquet image order changed")
            if any(hashlib.sha256(v["bytes"]).hexdigest() != report["image_sha256"][v["path"]] for v in images):
                raise ValueError("Parquet image bytes changed")
            checked["records"] += 1
            checked["images"] += len(images)
        files.append({"path": str(path.relative_to(output)), "questions": len(batch), "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = build_manifest(Path(source).name, sha256(source), rows, report, files, checked)
    atomic_json(output / "release_manifest.json", manifest)
    (output / "README.md").write_text(render_card(manifest), encoding="utf-8")
    return manifest


def build_manifest(source_name, source_sha256, rows, media, shards, checked):
    """Release manifest: counts, shard checksums and the validation actually performed.

    `checked` counts the records and images compared after writing; export raises
    on any mismatch, so the flags are true only when every row was compared.
    """
    images = sum(len(r["image_paths"]) for r in rows)
    complete = checked["records"] == len(rows) and checked["images"] == images
    return {
        "annotation_source": source_name, "source_sha256": source_sha256, "split": "test",
        "questions": len(rows), "scenes": len({(r["dataset"], r["scene_id"]) for r in rows}),
        "unique_images": media["unique_images"], "max_images_per_question": max(len(r["image_paths"]) for r in rows),
        "by_level": dict(sorted(Counter(r["level"] for r in rows).items())),
        "by_type": dict(sorted(Counter(f"{r['level']} / {r['type']}" for r in rows).items())),
        "by_source_type": dict(sorted(Counter(r["source_type"] for r in rows).items())),
        "type_name_mapping": [{"source_type": code, "level": LEVELS[code], "type": name,
                               "questions": sum(r["source_type"] == code for r in rows)}
                              for code, name in TYPE_NAMES.items()],
        "view_roles": {"multi_image_order": ["main_a", "bridge (zero or more)", "main_b"],
                       "single_image_role": "image", "single_image_a_b_names": None,
                       "role_occurrences": dict(Counter(role for r in rows for role in r["image_roles"])),
                       "image_count_distribution": {str(k): v for k, v in
                                                    sorted(Counter(len(r["image_paths"]) for r in rows).items())}},
        "parquet_shards": shards,
        "validation": {
            "method": ("Each serialized record is compared with its source record: source_record_json equals the "
                       "source record, every annotation column equals its conversion from that record, image order "
                       "equals image_paths, and each image's sha256 equals media.image_sha256."),
            "questions_checked": checked["records"], "images_checked": checked["images"],
            "all_source_records_equal": complete, "all_columns_match_source": complete,
            "all_image_bytes_and_order_unchanged": complete,
            "level_type_categories": len({f"{r['level']} / {r['type']}" for r in rows}),
            "distinct_display_names": len({r["type"] for r in rows}),
        },
        "media": media,
    }


def render_card(manifest):
    levels = "; ".join(f"{level}: {count:,}" for level, count in manifest["by_level"].items())
    return DATA_CARD.format(questions=manifest["questions"], scenes=manifest["scenes"],
                            categories=manifest["validation"]["level_type_categories"], levels=levels,
                            unique_images=manifest["unique_images"], max_images=manifest["max_images_per_question"])


DATA_CARD = """---
task_categories:
- visual-question-answering
language:
- en
tags:
- spatial-reasoning
- multi-image
pretty_name: SpaceCast-Bench
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*.parquet
---
# SpaceCast-Bench

An evaluation-only benchmark of {questions:,} multiple-choice spatial-reasoning
questions over {scenes} indoor scenes from ScanNet and ScanNet++, in {categories}
categories ({levels}). Each question has 1–{max_images} images; the {unique_images:,}
distinct images are embedded as JPEG/PNG bytes in `images: List(Image())`, so
downloading the Parquet files includes them. There is only a `test` split, and
no training data or model weights.

```python
from datasets import load_dataset
ds = load_dataset("hongxingli/SpaceCast-Bench", split="test")
images = ds[0]["images"]  # ordered PIL images
```

`question_uid` identifies each question. `type` and `question_type` use the
task names in the paper tables, and `source_type` holds the corresponding type
code. L2 and L3 both contain `Rotate Object Direction`; use `level` or
`source_type` to distinguish them, and group results by `(level, type)` to keep
all {categories} categories. `answer` is a list of option letters: one for
single-choice questions, all correct letters for multiple-choice questions,
which `multi_select` identifies. `source_record_json` holds the complete
annotation record.

For multi-image questions, `images` is ordered **Main view A → Bridge views →
Main view B**. The first and last images are main views; only intermediate
images are bridges. Two-image questions contain A/B with no bridges. A
single-image question is labeled simply `Image 1`, with no A/B designation.

| Column | Meaning |
| --- | --- |
| `main_image_a_name` | First main view name for multi-image questions; null for a single image |
| `main_image_b_name` | Last main view name for multi-image questions; null for a single image |
| `bridge_view_names` | Intermediate image names in viewing order; empty for one/two images |
| `image_roles` | Roles aligned with `images`: `main_a`, zero or more `bridge`, `main_b`; `["image"]` for one image |
| `image_name` | Name of the first image |
| `auxiliary_image_names` | Names of the remaining images in order, ending with Main view B |

`images` follows `[image_name] + auxiliary_image_names` and stores the original
encoded files. Attachment questions are L1. Single-choice scoring compares the
letter; multiple-choice scoring requires exact set equality. Missing, invalid
and failed predictions count as incorrect in the full denominator. The
evaluation code and complete scoring rules are in the accompanying code
repository.

All images originate from ScanNet and ScanNet++, and their original access and
use terms apply; the Apache-2.0 code license does not grant image redistribution
rights. No separate annotation license is asserted here. `release_manifest.json`
records exact counts, shard checksums and validation results.

## Citation

```bibtex
@inproceedings{{spacecastbench,
  title     = {{[PAPER TITLE]}},
  author    = {{[AUTHORS]}},
  booktitle = {{[VENUE]}},
  year      = {{[YEAR]}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, help="Source annotation JSONL")
    parser.add_argument("--image-root", required=True, help="Directory containing dataset/scene/frame.jpg")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-report", required=True, help="Written even when images are incomplete")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--shard-rows", type=int, default=100)
    args = parser.parse_args()
    if args.audit_only:
        rows = [normalize(row) for row in read_rows(args.source_jsonl)]
        report = audit_media(rows, args.image_root)
        atomic_json(args.audit_report, report)
        print(json.dumps({k: v for k, v in report.items() if k not in {"image_sha256", "missing", "invalid"}}, indent=2))
        if report["missing"] or report["invalid"]:
            raise SystemExit(2)
    else:
        result = export(args.source_jsonl, args.image_root, args.output_dir, shard_rows=args.shard_rows, audit_path=args.audit_report)
        print(json.dumps({k: v for k, v in result.items() if k not in {"media", "parquet_shards"}}, indent=2))


if __name__ == "__main__":
    main()
