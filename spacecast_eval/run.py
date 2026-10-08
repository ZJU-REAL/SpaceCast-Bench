"""Run model inference through an API, then score without changing gold annotations."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from .data import read_rows
from .loading import add_data_arguments, data_source, decode_images, fingerprint, from_args, gold_rows, image_bytes
from .protocol import MODE_RULES, MODES, PROTOCOL, build_prompt, check_mode, parse_response, prepare_images, protocol_hash
from .scoring import atomic_json, environment, evaluation_scope, write_report


class RunAborted(RuntimeError):
    """Raised when every recent call failed, so continuing would only record errors."""


class EmptyResponseError(RuntimeError):
    """The model returned no text."""


class NoAnswerError(RuntimeError):
    """direct mode: the response contained no parseable option."""


def retry_wait(error, attempt, retry_delay):
    # Rate limits back off from 60 s; other failures from retry_delay. Both double per attempt.
    text = str(error)
    return (60.0 if "429" in text or "quota" in text.lower() else retry_delay) * (2 ** attempt)


def ask(backend, prompt, blobs, row, mode, *, retries, retry_delay, sleep=time.sleep):
    """One question, retried as in the paper's evaluator. Returns (response, error)."""
    error = None
    for attempt in range(retries + 1):
        try:
            response = (backend.generate(prompt, blobs) or "").strip()
            if not response:
                raise (NoAnswerError if MODE_RULES[mode]["retry_unparsed"] else EmptyResponseError)("empty response")
            if MODE_RULES[mode]["retry_unparsed"] and parse_response(response, row) is None:
                raise NoAnswerError("response did not contain a parseable final option")
            return response, None
        except Exception as e:
            error = e
            if attempt < retries:
                sleep(retry_wait(e, attempt, retry_delay))
    return None, error


def run(ds, backend_factory, output_dir, settings, *, mode, image_root=None, limit=None, resume=False,
        retry_errors=False, retries=2, retry_delay=2.0, max_consecutive_errors=5, metadata=None):
    check_mode(mode)
    if retries < 0 or retry_delay < 0:
        raise ValueError("retries and retry_delay must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {"mode": mode, "protocol": PROTOCOL, "protocol_sha256": protocol_hash(),
              "retries": retries,
              "dataset_fingerprint": fingerprint(ds), "settings": settings,
              "image_root": str(Path(image_root).resolve()) if image_root else None}
    config_path, prediction_path = output / "run_config.json", output / "predictions.jsonl"
    if any(output.iterdir()) and not resume:
        raise ValueError("Output directory is not empty; use --resume or a new directory")
    if resume and any(output.iterdir()):
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume requires the same dataset, model and generation settings")
    else:
        atomic_json(config_path, config)
    gold = gold_rows(ds)
    by_id = {row["question_uid"]: row for row in gold}
    previous = read_rows(prediction_path) if prediction_path.exists() else []
    saved = {}
    for pred in previous:
        uid = pred["question_uid"]
        if uid not in by_id or uid in saved:
            raise ValueError(f"Unknown or duplicate saved UID: {uid}")
        saved[uid] = pred
    count = min(limit, len(ds)) if limit else len(ds)
    selected = gold[:count]
    needed = [i for i, row in enumerate(selected) if row["question_uid"] not in saved or
              (retry_errors and saved[row["question_uid"]].get("status") == "error")]
    # Validate all required media before instantiating a paid API/GPU backend.
    for i in needed:
        decode_images(prepare_images(image_bytes(ds[i], image_root)))
    backend = backend_factory() if needed else None
    from tqdm.auto import tqdm
    failures = []
    for i in tqdm(needed, desc="SpaceCast evaluation"):
        row = ds[i]
        uid = row["question_uid"]
        prompt = build_prompt(row, mode)
        blobs = prepare_images(image_bytes(row, image_root))
        result = {"question_uid": uid, "raw_response": None, "prediction": None, "status": "error",
                  "image_count": len(blobs)}
        response, error = ask(backend, prompt, blobs, row, mode, retries=retries, retry_delay=retry_delay)
        if error is None:
            parsed = parse_response(response, row)
            result.update(raw_response=response, prediction=parsed, status="valid" if parsed is not None else "invalid")
        else:
            # Do not persist SDK exception bodies containing requests or credentials;
            # the type and HTTP status are enough to diagnose a misconfigured backend.
            result["error_type"] = "invalid_direct_response" if isinstance(error, NoAnswerError) else type(error).__name__
            status_code = getattr(error, "status_code", None)
            if isinstance(status_code, int):
                result["error_status"] = status_code
        saved[uid] = result
        tmp = prediction_path.with_name("predictions.jsonl.tmp")
        tmp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in saved.values()), encoding="utf-8")
        tmp.replace(prediction_path)
        # A model that answers without a parseable option is not a broken backend.
        call_failed = result["status"] == "error" and result["error_type"] != "invalid_direct_response"
        failures = failures + [result] if call_failed else []
        if max_consecutive_errors and len(failures) >= max_consecutive_errors:
            kinds = Counter(f"{r['error_type']} ({r['error_status']})" if "error_status" in r else r["error_type"]
                            for r in failures)
            summary = ", ".join(f"{kind} x{n}" for kind, n in kinds.most_common())
            raise RunAborted(f"Stopped after {len(failures)} consecutive failed calls: {summary}. "
                             "Completed predictions are saved; fix the backend settings, then rerun "
                             "with --resume --retry-errors.")
    selected_ids = {r["question_uid"] for r in selected}
    scope, check = evaluation_scope(gold, count)
    # A subset run must not advertise the full denominator, or a smoke test reads
    # as a complete benchmark result. Full size is recorded separately.
    return write_report(output, selected, [p for uid, p in saved.items() if uid in selected_ids],
                        {"mode": mode, "protocol_sha256": config["protocol_sha256"], "dataset_fingerprint": config["dataset_fingerprint"],
                         "dataset_questions": len(selected), "benchmark_questions": len(ds),
                         "evaluation_scope": scope, "release_check": check, "settings": settings,
                         **(metadata or {})})


def main():
    from .backends import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT, PROVIDERS, THINKING_CHOICES
    parser = argparse.ArgumentParser(
        description=__doc__ + " Every default reproduces the setting of the paper's runs.")
    add_data_arguments(parser)
    parser.add_argument("--mode", choices=MODES, required=True,
                        help="cot: brief reasoning, then a final answer line (main experiment); "
                             "direct: the answer only. Both use the same parser and image preprocessing")
    parser.add_argument("--api-provider", choices=PROVIDERS, default="openai_chat",
                        help="openai_chat: any OpenAI-compatible server (vLLM, SGLang, hosted APIs); "
                             "anthropic: the Anthropic Messages API (used for Claude in the paper)")
    parser.add_argument("--model", required=True, help="Model name as served by the endpoint")
    parser.add_argument("--base-url", default=None,
                        help="API endpoint; required unless OPENAI_BASE_URL / ANTHROPIC_BASE_URL is set")
    parser.add_argument("--api-key-env", default=None,
                        help="Environment variable holding the key (default OPENAI_API_KEY, or ANTHROPIC_API_KEY "
                             "for anthropic; a name ending in AUTH_TOKEN is sent as a bearer token)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Output token limit; the paper's value per model is listed in the README")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature (default 0, greedy). Not sent to GPT-5, o-series, Claude 4 "
                             "and Claude Sonnet 5 model names, which reject it")
    parser.add_argument("--no-temperature", action="store_true", help="Never send a temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Sent only when set (not used in the paper)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Sent only when set (not used in the paper); extra_body.top_k for openai_chat")
    parser.add_argument("--seed", type=int, default=None, help="openai_chat only; sent only when set (not used in the paper)")
    parser.add_argument("--thinking", choices=THINKING_CHOICES, default="default",
                        help="openai_chat only. default: keep the model's own thinking default (model names "
                             "containing Qwen3.5 get enable_thinking on for cot, off for direct); disabled / "
                             "enabled: send chat_template_kwargs.enable_thinking (vLLM / SGLang)")
    parser.add_argument("--token-param", choices=["max_tokens", "max_completion_tokens"], default=None,
                        help="openai_chat only: request field carrying --max-tokens (default: "
                             "max_completion_tokens for GPT-5 and o-series model names, otherwise max_tokens)")
    parser.add_argument("--image-root", help="For data without embedded images (e.g. --jsonl): directory containing dataset/scene/frame.jpg")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test the first N questions; report is explicitly a subset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="Retry previously failed calls when resuming")
    parser.add_argument("--retries", type=int, default=2,
                        help="Extra attempts per question after a failed call, an empty response or, in direct "
                             "mode, a response without a parseable option")
    parser.add_argument("--retry-delay", type=float, default=2.0,
                        help="Initial wait in seconds before a retry; doubles per attempt (60 s for rate limits)")
    parser.add_argument("--max-consecutive-errors", type=int, default=5,
                        help="Stop after this many consecutive failed calls; 0 never stops")
    args = parser.parse_args()
    if (args.max_tokens < 1 or args.timeout <= 0 or args.max_consecutive_errors < 0 or args.retries < 0
            or args.retry_delay < 0 or args.temperature < 0 or (args.top_p is not None and not 0 < args.top_p <= 1)
            or (args.top_k is not None and args.top_k < 1)):
        parser.error("Invalid generation, timeout or error-limit settings")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    anthropic = args.api_provider == "anthropic"
    if anthropic and (args.thinking != "default" or args.seed is not None or args.token_param):
        parser.error("--thinking, --seed and --token-param apply only to --api-provider openai_chat")
    from .backends import APIBackend, is_reasoning_chat_model, should_omit_temperature, thinking_body
    # No silent default endpoint: a wrong provider fails every call.
    base_url = args.base_url or os.environ.get("ANTHROPIC_BASE_URL" if anthropic else "OPENAI_BASE_URL")
    if not base_url:
        parser.error("--base-url (or OPENAI_BASE_URL / ANTHROPIC_BASE_URL) is required")
    api_key_env = args.api_key_env or ("ANTHROPIC_API_KEY" if anthropic else "OPENAI_API_KEY")
    token_param = "max_tokens" if anthropic else (
        args.token_param or ("max_completion_tokens" if is_reasoning_chat_model(args.model) else "max_tokens"))
    omit_temperature = args.no_temperature or should_omit_temperature(args.model)
    settings = {"api_provider": args.api_provider, "model": args.model, "base_url": base_url,
                "api_key_env": api_key_env, "timeout": args.timeout, "max_tokens": args.max_tokens,
                "token_param": token_param, "temperature": None if omit_temperature else args.temperature,
                "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed, "thinking": args.thinking,
                "thinking_request": {} if anthropic else thinking_body(args.thinking, args.model, args.mode)}
    factory = lambda: APIBackend(
        args.model, base_url, args.mode, provider=args.api_provider, api_key_env=api_key_env,
        max_tokens=args.max_tokens, timeout=args.timeout, token_param=None if anthropic else token_param,
        temperature=args.temperature, omit_temperature=omit_temperature, top_p=args.top_p, top_k=args.top_k,
        seed=args.seed, thinking=args.thinking)
    packages = ["datasets", "pyarrow", "Pillow", "anthropic" if anthropic else "openai"]
    try:
        report = run(from_args(args), factory, args.output_dir, settings, mode=args.mode, image_root=args.image_root,
                     limit=args.limit, resume=args.resume, retry_errors=args.retry_errors,
                     retries=args.retries, retry_delay=args.retry_delay,
                     max_consecutive_errors=args.max_consecutive_errors,
                     metadata={"data_source": data_source(args), "environment": environment(packages)})
    except RunAborted as e:
        parser.exit(1, f"error: {e} Check --base-url, the key, --token-param and --no-temperature.\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
