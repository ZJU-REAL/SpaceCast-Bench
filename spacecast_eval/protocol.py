"""Evaluation protocol of the paper's experiments, shared by the cot and direct modes.

Both modes use the same system prompt, question/option layout, image
preprocessing, message order and answer parser. They differ only where the
paper's evaluator differs, as listed in MODE_RULES: the instruction appended to
the prompt, whether separately returned reasoning text is kept, whether a
response without a parseable answer is requested again, and the thinking flag
sent to models named Qwen3.5. Request settings (temperature, token limit,
thinking) live in the backend, with the paper's values as defaults.
"""
import hashlib
import io
import json
import re
from collections import defaultdict

PROTOCOL = "spacecast-v1"
MODES = ("cot", "direct")

SYSTEM_PROMPT = (
    "You are a careful vision-language assistant solving multiple-choice "
    "spatial reasoning questions about an image."
)

PROMPT_SUFFIX = (
    "Work through your reasoning briefly, then give your final choice as the LAST line.\n"
    "Keep your reasoning as short as possible (a few sentences at most) so the final Answer line is not truncated.\n"
    "Return this format:\n"
    "Reasoning: <brief reasoning>\n"
    "Answer: <single letter>"
)

MULTI_SELECT_PROMPT_SUFFIX = (
    "Work through your reasoning briefly, then give your final choice as the LAST line.\n"
    "If more than one option is correct, list all letters comma-separated.\n"
    "Keep your reasoning as short as possible (a few sentences at most) so the final Answer line is not truncated.\n"
    "Return this format:\n"
    "Reasoning: <brief reasoning>\n"
    "Answer: <letter(s)>"
)

DIRECT_PROMPT_SUFFIX = "Answer with a single letter only (A, B, C, or D). Do not explain."

DIRECT_MULTI_SELECT_PROMPT_SUFFIX = (
    "Answer with the correct letter(s) only, comma-separated if more than one. Do not explain."
)

MODE_RULES = {
    "cot": {"suffix": PROMPT_SUFFIX, "multi_select_suffix": MULTI_SELECT_PROMPT_SUFFIX,
            # Reasoning returned in a separate field is kept, ahead of the answer.
            "include_reasoning": True,
            "retry_unparsed": False,
            "qwen35_enable_thinking": True},
    "direct": {"suffix": DIRECT_PROMPT_SUFFIX, "multi_select_suffix": DIRECT_MULTI_SELECT_PROMPT_SUFFIX,
               # Only the final answer content is used.
               "include_reasoning": False,
               # A response without a parseable option is requested again, and is
               # recorded as an error once the retries are used up.
               "retry_unparsed": True,
               "qwen35_enable_thinking": False},
}

# Every image is re-encoded as JPEG with its longest side capped at this size.
IMAGE_MAX_PX = 1280
IMAGE_JPEG_QUALITY = 90


def check_mode(mode):
    if mode not in MODE_RULES:
        raise ValueError(f"Unknown evaluation mode {mode!r}; expected one of {', '.join(MODES)}")
    return mode


def build_prompt(row, mode):
    rules = MODE_RULES[check_mode(mode)]
    parts = [str(row.get("question") or "").strip(), ""]
    for idx, option in enumerate(row.get("options") or []):
        parts.append(f"{chr(65 + idx)}) {option}")
    parts.extend(["", rules["multi_select_suffix"] if row.get("multi_select") else rules["suffix"]])
    return "\n".join(parts)


def prepare_image(blob):
    from PIL import Image
    with Image.open(io.BytesIO(blob)) as image:
        image = image.convert("RGB")
        image.thumbnail((IMAGE_MAX_PX, IMAGE_MAX_PX))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=IMAGE_JPEG_QUALITY)
    return out.getvalue()


def prepare_images(blobs):
    return [prepare_image(blob) for blob in blobs]


def parse_prediction(value, allowed, multi):
    """Validate an explicit prediction (a letter, a comma list or a list of letters)."""
    if isinstance(value, str):
        value = value.strip().upper()
        if not re.fullmatch(r"[A-Z](?:\s*[,;]\s*[A-Z])*", value):
            return None
        letters = re.split(r"\s*[,;]\s*", value)
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        letters = [v.strip().upper() for v in value]
    else:
        return None
    if (not letters or len(set(letters)) != len(letters) or
            not set(letters) <= set(allowed) or (not multi and len(letters) != 1)):
        return None
    return frozenset(letters)


def allowed_letters(row):
    n_options = len(row.get("options") or [])
    if n_options <= 0:
        return "ABCD"
    return "".join(chr(65 + idx) for idx in range(min(n_options, 26)))


def _ordered_unique_letters(values, letters):
    seen = {value.upper() for value in values if value and value.upper() in letters.upper()}
    return [letter for letter in letters.upper() if letter in seen]


def _parse_multi_candidate(candidate, letters):
    allowed = re.escape(letters.upper())
    tokens = re.findall(rf"(?<![A-Z0-9])([{allowed}])(?![A-Z0-9])", candidate)
    if tokens:
        return _ordered_unique_letters(tokens, letters)
    compact = re.sub(r"[\s,;/&+|\-]+", "", candidate)
    if compact and re.fullmatch(rf"[{allowed}]+", compact):
        return _ordered_unique_letters(list(compact), letters)
    return []


def parse_answers(raw, letters):
    if not raw:
        return []
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()

    parsed = _parse_multi_candidate(upper, letters)
    if parsed and re.fullmatch(rf"[\s{allowed},;/&+|\-]+", upper):
        return parsed

    tail = re.search(
        rf"(?:^|[\r\n.!?])\s*([\(\[]?[{allowed}]"
        rf"(?:\s*(?:[,;/&+|\-]|\bAND\b)\s*[{allowed}])*\s*[\)\]]?)\s*$",
        upper,
    )
    if tail:
        parsed = _parse_multi_candidate(tail.group(1), letters)
        if parsed:
            return parsed

    patterns = [
        r"(?:FINAL\s+)?ANSWER(?:S)?(?:\s+IS|\s+ARE)?\s*[:：-]?\s*([^\r\n]+)",
        r"(?:CHOICES?|OPTIONS?)(?:\s+IS|\s+ARE)?\s*[:：-]?\s*([^\r\n]+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, upper))
        if matches:
            parsed = _parse_multi_candidate(matches[-1].group(1), letters)
            if parsed:
                return parsed
    return []


def _normalized_option_text(value):
    return re.sub(r"[\W_]+", " ", str(value or "").casefold()).strip()


def _parse_explicit_option_text(raw, options):
    if not options:
        return None
    option_keys = defaultdict(list)
    for index, option in enumerate(options):
        key = _normalized_option_text(option)
        if key:
            option_keys[key].append(chr(65 + index))

    stripped = raw.strip()
    last_line = next(
        (line.strip() for line in reversed(stripped.splitlines()) if line.strip()),
        "",
    )
    candidates = [stripped] if "\n" not in stripped and "\r" not in stripped else []
    patterns = [
        r"^(?:[-*]\s*)?\**(?:final\s+)?(?:answer|choice|option)\**"
        r"(?:\s+(?:is|would\s+be|should\s+be))?\s*[:：-]?\s*(.+?)\s*$",
        r"^(?:[-*]\s*)?\**(?:the|my)\s+(?:final\s+)?(?:answer|choice|option)\**"
        r"\s+is\s*[:：-]?\s*(.+?)\s*$",
        r"^(?:[-*]\s*)?(?:i(?:'ll|\s+will|\s+would)?\s+)?"
        r"(?:choose|select|go\s+with)\s*[:：-]?\s*(.+?)\s*$",
        r"^(?:[-*]\s*)?(?:so|therefore|thus|hence|in\s+conclusion)"
        r"\s*[:,]?\s*(.+?)\s*$",
        r"^(?:[-*]\s*)?(.+?)\s+is\s+(?:the|my)\s+(?:final\s+)?answer\s*[.!]?\s*$",
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, last_line, flags=re.IGNORECASE)
        if match:
            candidates.append(match.group(1))

    matched_letters = set()
    for candidate in candidates:
        letters_for_key = option_keys.get(_normalized_option_text(candidate), [])
        if len(letters_for_key) == 1:
            matched_letters.add(letters_for_key[0])
    return next(iter(matched_letters)) if len(matched_letters) == 1 else None


def parse_answer(raw, letters, options=None):
    if not raw:
        return None
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()
    if re.fullmatch(rf"[{allowed}]", upper):
        return upper

    # Some models append the final letter to the last punctuation mark after
    # emitting reasoning (for example, "...sense.B").
    tail = re.search(rf"(?:^|[\r\n.!?])\s*[\(\[]?([{allowed}])[\)\].!?]?\s*$", upper)
    if tail:
        return tail.group(1)

    patterns = [
        rf"(?:FINAL\s+)?ANSWER(?:\s+IS)?\s*[:：-]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"(?:CHOICE|OPTION)(?:\s+IS)?\s*[:：-]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"^[\(\[]?\s*([{allowed}])\s*[\)\].:：-]",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, upper))
        if matches:
            return matches[-1].group(1)
    return _parse_explicit_option_text(raw, options)


def parse_response(text, row):
    """Sorted predicted letters, or None when no answer can be recovered. Same in both modes."""
    if not isinstance(text, str):
        return None
    letters = allowed_letters(row)
    if row.get("multi_select"):
        predictions = parse_answers(text, letters)
        return sorted(predictions) if predictions else None
    prediction = parse_answer(text, letters, row.get("options"))
    return [prediction] if prediction else None


_PROBE_ROWS = [
    {"question": "Which option?", "options": ["first", "second", "third"], "multi_select": False},
    {"question": "Which options?", "options": ["first", "second", "third", "fourth"], "multi_select": True},
]
_PROBE_RESPONSES = ["B", "b", " A , C ", "A;C", "Answer: B", "answer: c,a", "Reasoning: x.\nAnswer: B",
                    "Answer: B.", "**Answer: B**", "Answer: (B)", "Answer: A and C", "The answer is B",
                    "Answer: A\nMaybe B", "Answer: B,B", "Answer: E", "", "A or B", "...sense.B",
                    "Answer: second", "So third", "Reasoning only."]


def protocol_hash():
    """Hash of what the protocol does, not of how its source is spelled.

    Covers both modes' prompts and mode rules, image preprocessing, the
    OpenAI-compatible and Anthropic message structures and parser decisions on a
    fixed probe set.
    Comments, formatting and line endings leave it unchanged; any change to
    model input or answer parsing changes it.
    """
    from PIL import Image
    from .backends import anthropic_messages, api_messages

    def encoded(fmt, size=(2, 2)):
        out = io.BytesIO()
        Image.new("RGB", size).save(out, format=fmt)
        return out.getvalue()

    def structure(content):
        # Image payloads are data, not protocol; keep only their position and MIME type.
        if isinstance(content, str):
            return content
        return [item["text"] if item["type"] == "text" else
                item["image_url"]["url"].split(",", 1)[0] if item["type"] == "image_url" else
                item["source"]["media_type"] for item in content]

    def messages(items):
        return [{**message, "content": structure(message["content"])} for message in items]

    with Image.open(io.BytesIO(prepare_image(encoded("PNG", (2000, 1500))))) as prepared:
        preprocessing = [prepared.format, prepared.size]
    blobs = prepare_images([encoded("JPEG"), encoded("PNG"), encoded("JPEG")])
    payload = {
        "protocol": PROTOCOL,
        "mode_rules": MODE_RULES,
        "prompts": {mode: [build_prompt(row, mode) for row in _PROBE_ROWS] for mode in MODES},
        "image_preprocessing": preprocessing,
        "api_messages": [messages(api_messages("PROMPT", blobs[:n])) for n in range(1, 4)],
        "anthropic_messages": [messages(anthropic_messages("PROMPT", blobs[:n])) for n in range(1, 4)],
        "parsing": [[parse_response(text, row) for text in _PROBE_RESPONSES] for row in _PROBE_ROWS],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
