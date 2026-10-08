"""API inference through an OpenAI-compatible or the Anthropic Messages API."""
import base64
import io
import os
from .protocol import MODE_RULES, SYSTEM_PROMPT, check_mode

PROVIDERS = ("openai_chat", "anthropic")
# Defaults of the paper's evaluator; each can be changed on the command line.
DEFAULT_MAX_TOKENS = 3072
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 60.0


def encode_images(blobs):
    from PIL import Image
    encoded = []
    for blob in blobs:
        with Image.open(io.BytesIO(blob)) as image:
            mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(image.format)
            if not mime:
                raise ValueError("Unsupported image format")
        encoded.append((base64.b64encode(blob).decode("ascii"), mime))
    return encoded


def api_content(prompt, blobs):
    """OpenAI-compatible user content: the prompt, then the images in order, without labels."""
    return [{"type": "text", "text": prompt},
            *({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}} for data, mime in encode_images(blobs))]


def api_messages(prompt, blobs):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": api_content(prompt, blobs)}]


def anthropic_messages(prompt, blobs):
    """Anthropic user content: the images in order, then the prompt (the system prompt is a separate field)."""
    content = [{"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
               for data, mime in encode_images(blobs)]
    return [{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}]


def is_reasoning_chat_model(model):
    """GPT-5 / o-series chat models take max_completion_tokens and only the default temperature."""
    return (model or "").lower().startswith(("gpt-5", "o1", "o3", "o4"))


def should_omit_temperature(model):
    """Models or proxies that reject an explicit temperature."""
    name = (model or "").lower()
    return is_reasoning_chat_model(model) or name.startswith(
        ("claude-opus-4", "claude-sonnet-4", "claude-sonnet-5", "claude-4"))


# --thinking: "default" leaves the model's own thinking default alone, except that
# model names containing Qwen3.5 receive a top-level enable_thinking flag (on for
# cot, off for direct), as in the paper's evaluator. "disabled" / "enabled" send
# chat_template_kwargs.enable_thinking, read by vLLM / SGLang chat templates; the
# paper's vLLM-served Qwen3.5-2B/9B/27B runs used "disabled".
THINKING_CHOICES = ("default", "disabled", "enabled")


def supports_qwen_thinking_control(model):
    normalized = str(model).strip().lower().replace("_", "-")
    return "qwen3.5" in normalized or "qwen3-5" in normalized


def thinking_body(thinking, model, mode):
    if thinking == "disabled":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if thinking == "enabled":
        return {"chat_template_kwargs": {"enable_thinking": True}}
    if thinking != "default":
        raise ValueError(f"Unsupported thinking setting: {thinking}")
    if supports_qwen_thinking_control(model):
        return {"enable_thinking": MODE_RULES[mode]["qwen35_enable_thinking"]}
    return {}


def _get_field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _content_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(value)


def chunk_text(chunk, include_reasoning):
    """Text carried by one streamed chunk; reasoning_content only when include_reasoning."""
    if isinstance(chunk, str):
        return chunk
    fields = ("content", "reasoning_content", "text") if include_reasoning else ("content", "text")
    parts = []
    for choice in _get_field(chunk, "choices") or []:
        for container_name in ("delta", "message"):
            container = _get_field(choice, container_name)
            if container is None:
                continue
            for field in fields:
                text = _content_text(_get_field(container, field))
                if text:
                    parts.append(text)
        text = _content_text(_get_field(choice, "text"))
        if text:
            parts.append(text)
    return "".join(parts) or _content_text(_get_field(chunk, "content"))


class APIBackend:
    def __init__(self, model, base_url, mode, *, provider="openai_chat", api_key_env=None,
                 max_tokens=DEFAULT_MAX_TOKENS, timeout=DEFAULT_TIMEOUT, token_param=None,
                 temperature=DEFAULT_TEMPERATURE, omit_temperature=None, top_p=None, top_k=None, seed=None,
                 thinking="default"):
        """Every default reproduces the paper's runs. Optional sampling fields (top_p, top_k,
        seed) are sent only when set; `token_param` and `omit_temperature` default to the
        model-name rules in `is_reasoning_chat_model` and `should_omit_temperature`."""
        self.mode = check_mode(mode)
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported API provider: {provider}")
        if provider == "anthropic" and (thinking != "default" or seed is not None or token_param):
            raise ValueError("--thinking, --seed and --token-param apply only to the openai_chat provider")
        thinking_body(thinking, model, self.mode)
        self.provider, self.model, self.max_tokens, self.thinking = provider, model, max_tokens, thinking
        self.temperature, self.top_p, self.top_k, self.seed = temperature, top_p, top_k, seed
        self.token_param = token_param or ("max_completion_tokens" if is_reasoning_chat_model(model) else "max_tokens")
        if self.token_param not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"Unsupported token parameter: {self.token_param}")
        self.omit_temperature = should_omit_temperature(model) if omit_temperature is None else omit_temperature
        api_key_env = api_key_env or ("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"Set the API key in environment variable {api_key_env}")
        try:
            if provider == "anthropic":
                from anthropic import Anthropic
                # An *_AUTH_TOKEN variable is sent as a bearer token, as the paper's runs did.
                credential = {"auth_token": key} if api_key_env.upper().endswith("AUTH_TOKEN") else {"api_key": key}
                self.client = Anthropic(**credential, base_url=base_url, timeout=timeout)
            else:
                from openai import OpenAI
                self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        except ImportError as e:
            raise RuntimeError("Install the API dependencies: pip install -r requirements.txt") from e

    def request(self, prompt, blobs):
        if self.provider == "anthropic":
            kwargs = {"model": self.model, "system": SYSTEM_PROMPT, "messages": anthropic_messages(prompt, blobs),
                      "max_tokens": self.max_tokens}
            if not self.omit_temperature:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
            return kwargs
        kwargs = {"model": self.model, "messages": api_messages(prompt, blobs), self.token_param: self.max_tokens}
        if not self.omit_temperature:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.seed is not None:
            kwargs["seed"] = self.seed
        extra = thinking_body(self.thinking, self.model, self.mode)
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        if extra:
            kwargs["extra_body"] = extra
        # Streamed explicitly: some OpenAI-compatible proxies always reply with SSE.
        kwargs["stream"] = True
        return kwargs

    def generate(self, prompt, blobs):
        if self.provider == "anthropic":
            response = self.client.messages.create(**self.request(prompt, blobs))
            return "\n".join(str(block.text) for block in getattr(response, "content", None) or []
                             if getattr(block, "type", None) == "text" and getattr(block, "text", None))
        include_reasoning = MODE_RULES[self.mode]["include_reasoning"]
        stream = self.client.chat.completions.create(**self.request(prompt, blobs))
        return "".join(chunk_text(chunk, include_reasoning) for chunk in stream)
