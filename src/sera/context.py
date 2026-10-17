"""Bound tool output and fit chat history to the serving model's context."""

import json
import logging
from copy import deepcopy

import httpx

log = logging.getLogger(__name__)
MAX_READ_LINES = 200
MAX_TOOL_CHARS = 12000
NOTICE = (
    "[sera-cli: Older tool output or completed steps were removed to fit the "
    "context window. Files on disk are unchanged. Inspect git diff and reread "
    "targeted file ranges when earlier details are needed.]"
)


def clip_output(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + "\n[sera-cli: Output clipped. Request a smaller line range or a targeted search.]"
    )


class ContextBudgetError(ValueError):
    pass


async def fit_context(
    client, endpoint, headers, model, messages, tools, context_limit, output_tokens
):
    """Count with the serving tokenizer, then evict old observations/whole steps.

    Never remove user instructions or leave orphan tool results. The caller's
    history stays intact; only the copy sent to the backend is reduced.
    """
    history = deepcopy(messages)
    for message in history:
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            message["content"] = clip_output(message["content"])
    tokenizer_available = True
    tokenizer_url = endpoint.removesuffix("/v1/chat/completions") + "/tokenize"

    async def count():
        nonlocal tokenizer_available, context_limit
        payload = {
            "model": model,
            "messages": history,
            "tools": tools,
            "add_generation_prompt": True,
        }
        if tokenizer_available:
            try:
                response = await client.post(
                    tokenizer_url, json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()
                tokens = data["count"]
                if not isinstance(tokens, int) or tokens < 0:
                    raise ValueError("Invalid token count")
                backend_limit = data.get("max_model_len")
                if isinstance(backend_limit, int) and backend_limit > 0:
                    context_limit = min(context_limit, backend_limit)
                return tokens
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                tokenizer_available = False
                log.warning(
                    "Backend tokenization unavailable; using conservative UTF-8 byte budget"
                )
        # Conservative fallback for byte-based tokenizers, with template headroom.
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1024

    warned = False
    while True:
        tokens = await count()
        if tokens + output_tokens + 256 <= context_limit:
            return history, tokens
        if not warned:
            history.insert(0, {"role": "system", "content": NOTICE})
            warned = True
        # Evict the oldest substantial observation first. Keep recent results
        # unless removing older output/steps is insufficient.
        candidates = [
            m
            for m in history
            if m.get("role") == "tool" and len(m.get("content", "")) > 512
        ]
        if candidates:
            candidates[0]["content"] = (
                "[sera-cli: Earlier tool output omitted. Reread if needed.]"
            )
            continue
        # Drop one complete assistant step, including all its tool responses.
        # Preserve the latest assistant step, and every user/system message.
        starts = [i for i, m in enumerate(history) if m.get("role") == "assistant"]
        if len(starts) > 1:
            start = starts[0]
            end = start + 1
            while end < len(history) and history[end].get("role") == "tool":
                end += 1
            del history[start:end]
            continue
        raise ContextBudgetError(
            "Task instructions and the latest step exceed the model context budget. "
            "Use /clear and provide a shorter task or smaller file excerpts."
        )
