"""Gọi target model trên Modal — chỉ SDK hoặc HTTP endpoint."""
import time
import warnings

import requests

from config import (
    MAX_TOKENS,
    MODAL_APP_NAME,
    MODAL_ENDPOINT_URL,
    MODE,
    TARGET_TEMPERATURE,
    USE_RETRIEVAL,
    USE_TOOLS,
)

warnings.filterwarnings(
    "ignore",
    message=r".*WindowsSelectorEventLoopPolicy.*deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*asyncio\.set_event_loop_policy.*deprecated.*",
    category=DeprecationWarning,
)


def _call_sdk(messages, thinking_mode=False):
    import modal
    cls = modal.Cls.from_name(MODAL_APP_NAME, "LLMServer")
    t0 = time.time()
    kwargs = dict(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TARGET_TEMPERATURE,
        thinking_mode=thinking_mode,
        use_retrieval=USE_RETRIEVAL,
    )
    if USE_TOOLS is not None:
        kwargs["use_tools"] = USE_TOOLS
    res = cls().generate.remote(**kwargs)
    return res, time.time() - t0


def _call_http(messages, thinking_mode=False):
    if not MODAL_ENDPOINT_URL:
        raise ValueError("Thiếu MODAL_ENDPOINT_URL trong .env khi chạy HTTP")
    t0 = time.time()
    body = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TARGET_TEMPERATURE,
        "thinking_mode": thinking_mode,
        "use_retrieval": USE_RETRIEVAL,
    }
    if USE_TOOLS is not None:
        body["use_tools"] = USE_TOOLS
    r = requests.post(MODAL_ENDPOINT_URL, json=body, timeout=600)
    r.raise_for_status()
    return r.json(), time.time() - t0


def call_target(messages, thinking_mode=False):
    """Dispatch theo MODE trong config (`sdk` hoặc `http`)."""
    if MODE == "http":
        return _call_http(messages, thinking_mode)
    return _call_sdk(messages, thinking_mode)
