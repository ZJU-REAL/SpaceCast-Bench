"""Score predictions: exact letter/set match with a fixed selected-set denominator."""
import argparse
from collections import Counter, defaultdict
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform

from . import __version__
from .data import read_rows, release_check, sha256, source_type, TYPE_NAMES
from .loading import add_data_arguments, data_source, fingerprint, from_args, gold_rows
from .protocol import PROTOCOL, parse_prediction, parse_response, protocol_hash


def score(gold, predictions):
    by_id = {}
    for row in gold:
        uid = row["question_uid"]
        if uid in by_id:
            raise ValueError(f"Duplicate gold UID: {uid}")
        by_id[uid] = row
    if not by_id:
        raise ValueError("Empty gold set")
    pred_by_id = {}
    for row in predictions:
        uid = row["question_uid"]
        if uid in pred_by_id or uid not in by_id:
            raise ValueError(f"Duplicate or unknown prediction UID: {uid}")
        if "prediction" not in row and "raw_response" not in row:
            raise ValueError(f"Missing prediction/raw_response: {uid}")
        pred_by_id[uid] = row
    details = []
    error_types = Counter()
    for uid, row in by_id.items():
        code = source_type(row)
        allowed = [chr(65+i) for i in range(len(row["options"]))]
        multi = bool(row.get("multi_select", False))
        target = parse_prediction(row["answer"], allowed, multi)
        if target is None:
            raise ValueError(f"Invalid gold answer: {uid}")
        pred = pred_by_id.get(uid)
        parsed = None
        if pred is None:
            status = "missing"
        elif pred.get("status") == "error":
            status = "error"
            kind = pred.get("error_type") or "unknown"
            error_types[f"{kind} ({pred['error_status']})" if pred.get("error_status") else kind] += 1
        else:
            value = pred.get("prediction")
            # A JSON null is an absent prediction rather than an unparsable one,
            # so it falls through to raw_response instead of scoring as invalid.
            if value is None and pred.get("raw_response") is not None:
                value = parse_response(pred["raw_response"], row)
            parsed = parse_prediction(value, allowed, multi)
            status = "valid" if parsed is not None else "invalid"
        details.append({"question_uid": uid, "correct": parsed == target, "status": status,
                        "prediction": sorted(parsed) if parsed is not None else None, "answer": sorted(target),
                        "level": row["level"], "type": TYPE_NAMES[code], "source_type": code,
                        "dataset": row.get("dataset", "unknown")})

    def summarize(items):
        count = len(items)
        correct = sum(r["correct"] for r in items)
        return {"count": count, "correct": correct, "accuracy": correct/count,
                **{s: sum(r["status"] == s for r in items) for s in ("missing", "invalid", "error")}}

    report = {"protocol": PROTOCOL, "protocol_sha256": protocol_hash(),
              "overall": summarize(details), "prediction_coverage": len(pred_by_id)/len(by_id),
              "error_types": dict(sorted(error_types.items()))}
    for key in ("level", "dataset"):
        groups = defaultdict(list)
        for item in details:
            groups[item[key]].append(item)
        report[f"by_{key}"] = {k: summarize(v) for k, v in sorted(groups.items())}
    types = defaultdict(list)
    for item in details:
        types[(item["level"], item["type"])].append(item)
    # Exactly 16 level/type categories; same-name rotations must never merge.
    report["by_type"] = {f"{level} / {label}": summarize(v) for (level, label), v in sorted(types.items())}
    report["by_level_type"] = {}
    for (level, label), items in sorted(types.items()):
        report["by_level_type"].setdefault(level, {})[label] = summarize(items)
    report["macro_type_accuracy"] = sum(v["accuracy"] for v in report["by_type"].values()) / len(report["by_type"])
    return report, details


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def evaluation_scope(benchmark, scored):
    """Label a report "full" only for the complete published split.

    `benchmark` is every loaded gold row; `scored` is how many were scored. A
    --limit run is a "subset"; scoring all of a dataset that is not the release
    (a single shard, an edited copy) is "nonstandard".
    """
    check = release_check(benchmark)
    if scored < len(benchmark):
        return "subset", check
    return ("full" if check["matches_release"] else "nonstandard"), check


def environment(packages):
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return {"spacecast_eval": __version__, "python": platform.python_version(), "packages": versions}


def write_report(directory, gold, predictions, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report, details = score(gold, predictions)
    report.update(metadata or {})
    # Averaging only the categories a partial run happens to contain is not the
    # 16-category metric, so it is withheld rather than reported misleadingly.
    if report.get("evaluation_scope") != "full":
        report["macro_type_accuracy"] = None
    atomic_json(directory / "report.json", report)
    tmp = directory / "per_question.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details), encoding="utf-8")
    tmp.replace(directory / "per_question.jsonl")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--predictions", required=True, help="JSONL with question_uid and prediction or raw_response")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    ds = from_args(args)
    gold = gold_rows(ds)
    scope, check = evaluation_scope(gold, len(gold))
    report = write_report(args.output_dir, gold, read_rows(args.predictions),
                          {"dataset_fingerprint": fingerprint(ds),
                           "predictions_sha256": sha256(args.predictions),
                           "evaluation_scope": scope, "release_check": check,
                           "dataset_questions": len(gold), "benchmark_questions": len(gold),
                           "data_source": data_source(args), "environment": environment(["datasets", "pyarrow"])})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
