"""Lossless conversion of the evaluation annotations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

LEVELS = {
    "direction_agent": "L1", "direction_allocentric": "L1",
    "direction_object_centric": "L1", "distance": "L1", "occlusion": "L1",
    "attachment_chain": "L1", "object_move_agent": "L2",
    "object_move_allocentric": "L2", "object_move_object_centric": "L2",
    "object_move_distance": "L2", "object_move_occlusion": "L2",
    "object_remove": "L2", "object_rotate_object_centric": "L2",
    "coordinate_rotation_agent": "L3", "coordinate_rotation_allocentric": "L3",
    "coordinate_rotation_object_centric": "L3",
}

TYPE_NAMES = {
    "direction_agent": "Camera Direction",
    "direction_object_centric": "Object Direction",
    "direction_allocentric": "World Direction",
    "distance": "Distance",
    "occlusion": "Occlusion",
    "attachment_chain": "Attachment",
    "object_move_agent": "Move Camera Direction",
    "object_move_object_centric": "Move Object Direction",
    "object_rotate_object_centric": "Rotate Object Direction",
    "object_move_allocentric": "Move World Direction",
    "object_move_distance": "Move Distance",
    "object_move_occlusion": "Move Occlusion",
    "object_remove": "Remove Occlusion",
    "coordinate_rotation_agent": "Rotate Camera Direction",
    "coordinate_rotation_object_centric": "Rotate Object Direction",
    "coordinate_rotation_allocentric": "Rotate World Direction",
}


# The published test split. A run is only labelled "full" when its gold set
# matches all three; otherwise it is "nonstandard" (e.g. a single shard).
RELEASE_QUESTIONS = 3862
RELEASE_SCENES = 182
RELEASE_CATEGORY_COUNTS = {
    "L1 / Attachment": 97, "L1 / Camera Direction": 264, "L1 / Distance": 272,
    "L1 / Object Direction": 251, "L1 / Occlusion": 221, "L1 / World Direction": 263,
    "L2 / Move Camera Direction": 293, "L2 / Move Distance": 285,
    "L2 / Move Object Direction": 256, "L2 / Move Occlusion": 139,
    "L2 / Move World Direction": 264, "L2 / Remove Occlusion": 216,
    "L2 / Rotate Object Direction": 256, "L3 / Rotate Camera Direction": 263,
    "L3 / Rotate Object Direction": 256, "L3 / Rotate World Direction": 266,
}


def category(row):
    """Scoring category; level is part of the key so same-name L2/L3 tasks stay apart."""
    return f"{row['level']} / {TYPE_NAMES[source_type(row)]}"


def release_check(gold):
    counts = {}
    for row in gold:
        key = category(row)
        counts[key] = counts.get(key, 0) + 1
    scenes = {(row.get("dataset"), row.get("scene_id")) for row in gold}
    return {"questions": len(gold), "scenes": len(scenes), "categories": len(counts),
            "matches_release": (len(gold) == RELEASE_QUESTIONS and len(scenes) == RELEASE_SCENES
                                and counts == RELEASE_CATEGORY_COUNTS)}


def source_type(row):
    """Resolve paper labels using level; L2/L3 Rotate Object Direction differ."""
    label = row["type"]
    code = row.get("source_type")
    if code is None:
        if label in LEVELS:
            code = label
        else:
            matches = [k for k, name in TYPE_NAMES.items() if name == label and LEVELS[k] == row["level"]]
            if len(matches) != 1:
                raise ValueError(f"Unknown or ambiguous type/level: {row['question_uid']}")
            code = matches[0]
    if (code not in LEVELS or LEVELS[code] != row["level"] or
            label not in {code, TYPE_NAMES[code]} or
            (row.get("question_type") is not None and row["question_type"] != label)):
        raise ValueError(f"Inconsistent type/level/source_type: {row['question_uid']}")
    return code


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("Expected a JSON array or JSONL; inspect canonical JSON envelopes explicitly.")
    return obj


def safe_component(value):
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"Invalid path component: {value!r}")
    if PurePosixPath(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Not a basename: {value!r}")
    return value


def image_names(row):
    names = [row["image_name"]] + list(row.get("auxiliary_image_names") or [])
    if len(names) != len(set(names)):
        raise ValueError(f"Repeated image: {row['question_uid']}")
    for name in names:
        if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
                or "\\" in name or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError(f"Unsafe relative image name: {name!r}")
    return names


def view_metadata(names):
    """The ordered sequence is main A, zero or more bridges, then main B."""
    if not 1 <= len(names) <= 8:
        raise ValueError("Each question must have 1 to 8 images")
    return {
        "main_image_a_name": names[0] if len(names) > 1 else None,
        "main_image_b_name": names[-1] if len(names) > 1 else None,
        "bridge_view_names": names[1:-1],
        "image_roles": (["main_a"] + ["bridge"] * (len(names) - 2) + ["main_b"]
                        if len(names) > 1 else ["image"]),
    }


def validate_view_metadata(row, count=None):
    """Accept records without role fields; reject conflicting explicit roles or image counts."""
    if "image_name" not in row:
        return
    names = image_names(row)
    if count is not None and len(names) != count:
        raise ValueError(f"Image count disagrees with source names: {row['question_uid']}")
    for key, value in view_metadata(names).items():
        if key in row and row[key] != value:
            raise ValueError(f"Inconsistent {key}: {row['question_uid']}")


def normalize(row, default_dataset=None):
    uid = row["question_uid"]
    if not isinstance(uid, str) or not uid:
        raise ValueError("Missing question UID")
    dataset = row.get("dataset") or default_dataset
    if dataset not in {"scannet", "scannetpp"}:
        raise ValueError(f"Unknown source dataset for {uid}: {dataset!r}")
    code = source_type(row)
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= 26:
        raise ValueError(f"Invalid options: {uid}")
    labels = [chr(65 + i) for i in range(len(options))]
    answer = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
    multi = bool(row.get("multi_select", False))
    if (not answer or len(set(answer)) != len(answer) or
            not set(answer) <= set(labels) or (not multi and len(answer) != 1)):
        raise ValueError(f"Invalid answer: {uid}")
    values = [options[labels.index(a)] for a in answer]
    # Export sources carry a bare `correct_value` for single-choice rows; released
    # Parquet/HF rows carry only the `correct_values` list. Accept either rather
    # than requiring a key the published dataset does not have.
    source_values = row.get("correct_values")
    if not multi and "correct_value" in row:
        source_values = [row["correct_value"]]
    if (not isinstance(source_values, list) or len(source_values) != len(values)
            or set(source_values) != set(values)):
        raise ValueError(f"Answer/text mismatch: {uid}")
    scene = safe_component(row["scene_id"])
    names = image_names(row)
    return {
        "question_uid": uid, "dataset": dataset, "scene_id": scene,
        "level": row["level"], "type": TYPE_NAMES[code], "question_type": TYPE_NAMES[code],
        "source_type": code, "task": row["task"],
        "question": row["question"], "options": options, "option_labels": labels,
        "answer": answer, "multi_select": multi, "correct_values": values,
        "image_name": row["image_name"],
        "auxiliary_image_names": list(row.get("auxiliary_image_names") or []),
        **view_metadata(names),
        "image_paths": [f"{dataset}/{scene}/{n}" for n in names],
        "relation_unchanged": row.get("relation_unchanged"),
        "source_record_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
    }


def features():
    from datasets import Features, List, Value
    strings = ["question_uid", "dataset", "scene_id", "level", "type", "question_type", "source_type", "task",
               "question", "image_name", "main_image_a_name", "main_image_b_name", "source_record_json"]
    lists = ["options", "option_labels", "answer", "correct_values",
             "auxiliary_image_names", "bridge_view_names", "image_roles", "image_paths"]
    return Features({**{k: Value("string") for k in strings},
                     **{k: List(Value("string")) for k in lists},
                     "multi_select": Value("bool"), "relation_unchanged": Value("bool")})
