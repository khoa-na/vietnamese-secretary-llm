"""LLM-as-Judge: build prompt, call Gemini/DeepSeek/OpenRouter, sanitize verdict.

Public API:
  judge(case, output_text, system_prompt="") → dict
    {passed, overall, per_tc, reasoning, violations, judge_thinking}
"""
import json
import re
import time
from collections import deque
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_THINKING_ENABLED,
    GEMINI_API_KEY,
    JUDGE_MODEL,
    JUDGE_RPM,
    JUDGE_SEED,
    JUDGE_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    PASS_THRESHOLD,
)
from criteria import JUDGE_SCHEMA, JUDGE_SYSTEM, JUDGE_SYSTEM_RAG

# Nguồn RAG khai báo trong case bằng đường dẫn tương đối so với thư mục eval/.
_EVAL_DIR = Path(__file__).parent


# ───────────────────────────────────────────────
# Clients (lazy init, in module scope để chia sẻ rate-limit state)
# ───────────────────────────────────────────────
_gemini_client = None
if GEMINI_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo Gemini client: {e}")

# DeepSeek client: OpenAI-compatible API (base_url=https://api.deepseek.com).
_deepseek_client = None
if DEEPSEEK_API_KEY:
    try:
        _deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo DeepSeek client: {e}")

# OpenRouter client: OpenAI-compatible API for slugs like "qwen/qwen3.6-flash".
_openrouter_client = None
if OPENROUTER_API_KEY:
    try:
        headers = {}
        if OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = OPENROUTER_SITE_URL
        if OPENROUTER_APP_NAME:
            headers["X-Title"] = OPENROUTER_APP_NAME
        _openrouter_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers=headers or None,
        )
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo OpenRouter client: {e}")


# ───────────────────────────────────────────────
# Sliding-window rate limiter (đảm bảo ≤ JUDGE_RPM req/60s)
# ───────────────────────────────────────────────
_judge_call_times: "deque[float]" = deque()


def _rate_limit_wait():
    if JUDGE_RPM <= 0:
        return
    now = time.time()
    while _judge_call_times and now - _judge_call_times[0] >= 60.0:
        _judge_call_times.popleft()
    if len(_judge_call_times) >= JUDGE_RPM:
        wait = 60.0 - (now - _judge_call_times[0]) + 0.2
        if wait > 0:
            print(f" [rate-limit: sleep {wait:.1f}s] ", end="", flush=True)
            time.sleep(wait)
        now = time.time()
        while _judge_call_times and now - _judge_call_times[0] >= 60.0:
            _judge_call_times.popleft()
    _judge_call_times.append(time.time())


# ───────────────────────────────────────────────
# Context-aware must_not_contain checker
# ───────────────────────────────────────────────
# Phát hiện cụm cấm xuất hiện trong ngữ cảnh refusal/asking (bot nói "chưa có data"
# hoặc "vui lòng cung cấp X"). Nếu cụm cấm chỉ xuất hiện gần các marker này, đó là
# refusal hợp lệ, KHÔNG phải hallucination → KHÔNG override fail.
REFUSAL_MARKERS = [
    # Tiếng Việt — negation tokens cơ bản
    " không ", " chưa ", " chẳng ",
    # Tiếng Việt — phủ định / thiếu data
    "chưa có", "chưa nhận", "chưa được", "chưa truy cập", "chưa thể",
    "chưa biết", "chưa xác", "chưa rõ", "chưa ghi nhận", "chưa tìm",
    "không có", "không thể", "không nhận", "không truy cập", "không tìm",
    "không biết", "không xác", "không rõ", "không phải", "không trùng",
    # Tiếng Việt — asking / yêu cầu cung cấp
    "vui lòng cung cấp", "sếp cung cấp", "cần cung cấp", "xin cung cấp",
    "yêu cầu cung cấp", "hãy cung cấp", "cung cấp thêm", "cung cấp lại",
    "cho tôi biết", "có thể cho", "tránh nhầm", "tránh sai", "tránh bịa",
    # English
    "don't have", "do not have", "haven't", "have not received", "no access",
    "cannot confirm", "cannot verify", "cannot determine", "unable to",
    "please provide", "could you provide", "i don't know", "without",
    " no ", " not ", " never ",
]


def _phrase_regex(phrase_lower: str) -> "re.Pattern":
    """Khớp `phrase` theo RANH GIỚI TỪ để tránh false-match substring.

    Vd cụm cấm "dạ" KHÔNG được khớp bên trong "dạng". Chỉ đặt ranh giới ở
    phía nào của cụm là ký tự chữ/số (\\w bao gồm cả chữ tiếng Việt Unicode);
    phía là dấu câu (vd cụm "ạ.") thì không cần ranh giới.
    """
    esc = re.escape(phrase_lower)
    left = r"(?<!\w)" if phrase_lower[:1].isalnum() else ""
    right = r"(?!\w)" if phrase_lower[-1:].isalnum() else ""
    return re.compile(left + esc + right)


def _has_unsafe_assertion(text: str, phrase: str, strict: bool = False, window: int = 80) -> bool:
    """Trả True nếu `phrase` xuất hiện như một khẳng định (dấu hiệu hallucination).

    - strict=True (case `language: en`): bất kỳ occurrence nào cũng tính.
    - strict=False: chỉ tính occurrence KHÔNG nằm trong window ~80 ký tự cạnh
      một refusal marker.
    Khớp theo ranh giới từ (xem `_phrase_regex`) nên cụm ngắn không bị match
    nhầm bên trong một từ dài hơn.
    """
    phrase_lower = phrase.lower()
    if not phrase_lower:
        return False
    text_lower = text.lower()
    pattern = _phrase_regex(phrase_lower)
    for m in pattern.finditer(text_lower):
        if strict:
            return True
        ctx_start = max(0, m.start() - window)
        ctx_end = min(len(text_lower), m.end() + window)
        context = text_lower[ctx_start:ctx_end]
        if not any(mk in context for mk in REFUSAL_MARKERS):
            return True
    return False


# ───────────────────────────────────────────────
# Build prompt
# ───────────────────────────────────────────────
def build_judge_prompt(case, output_text, system_prompt=""):
    turns = case.get("turns", [])
    convo = "\n".join(f"[{t.get('role','user').upper()}] {t.get('content','')}" for t in turns)
    expected = case.get("expected", {}) or {}
    eval_notes = expected.get("eval_notes", "(không có)")
    must_contain = expected.get("must_contain", [])
    must_not_contain = expected.get("must_not_contain", [])
    criteria = case.get("criteria", [])
    criteria_str = ", ".join(criteria) if criteria else "(Không có chỉ định cụ thể)"

    extra = ""
    if must_contain:
        extra += f"\nGỢI Ý cần có (không bắt buộc keyword, chỉ cần ý): {must_contain}"
    if must_not_contain:
        extra += (
            f"\nCỤM TỪ CẤM (chỉ tính vi phạm khi xuất hiện như KHẲNG ĐỊNH, không tính nếu "
            f"nằm trong câu phủ định/refusal/asking): {must_not_contain}"
        )

    return f"""TEST CASE: {case.get('id')} — {case.get('name')}
Use case: {case.get('use_case')}
Loại: {case.get('type','?')} | Ngôn ngữ: {case.get('language','vi')}

SYSTEM PROMPT CỦA CHATBOT THƯ KÝ (Sử dụng để đối chiếu hành vi & văn phong giao tiếp):
\"\"\"
{system_prompt}
\"\"\"

CÁC TIÊU CHÍ ÁP DỤNG CHO CASE NÀY (xem định nghĩa chi tiết trong system prompt — đánh giá nghiêm ngặt theo các chiều này):
{criteria_str}

TIÊU CHÍ THÀNH CÔNG (EVAL NOTES):
{eval_notes}{extra}

HỘI THOẠI ĐẦU VÀO:
{convo}

CÂU TRẢ LỜI CỦA MODEL CẦN ĐÁNH GIÁ (Chú ý: Bỏ qua các khối suy nghĩ <thinking>...</thinking> bên trong phản hồi nếu có khi chấm các tiêu chí về độ dài dòng, văn phong hoặc định dạng, chỉ đánh giá phần văn bản hiển thị cho người dùng):
\"\"\"
{output_text}
\"\"\"

Hãy chấm theo schema JSON."""


# ───────────────────────────────────────────────
# RAG: resolve nguồn tài liệu truy xuất + build prompt RAG
# ───────────────────────────────────────────────
def _resolve_rag_source(case):
    """Trả về (kind, value):
      ('file', Path)  nếu `rag_source` trỏ tới file tồn tại → gửi file cho judge.
      ('text', str)   nếu `rag_source` là chuỗi text (nguồn nội tuyến).
      (None, None)    nếu không khai báo → nguồn nằm trong khối [Dữ liệu truy xuất] của hội thoại.
    """
    src = case.get("rag_source")
    if not src:
        return None, None
    p = _EVAL_DIR / str(src)
    if p.exists() and p.is_file():
        return "file", p
    return "text", str(src)


def build_judge_prompt_rag(case, output_text, system_prompt="", source_text=None, has_file=False):
    """Như build_judge_prompt nhưng thêm khối NGUỒN để judge đối chiếu grounding."""
    base = build_judge_prompt(case, output_text, system_prompt)
    if has_file:
        src_block = (
            "NGUỒN TÀI LIỆU TRUY XUẤT: đính kèm dưới dạng FILE — hãy ĐỌC TRỰC TIẾP nội dung file "
            "để đối chiếu mọi dữ kiện trong câu trả lời."
        )
    elif source_text:
        src_block = (
            "NGUỒN TÀI LIỆU TRUY XUẤT (đối chiếu câu trả lời với nguồn này):\n"
            f'"""\n{source_text}\n"""'
        )
    else:
        src_block = (
            "NGUỒN TÀI LIỆU TRUY XUẤT: nằm trong khối [Dữ liệu truy xuất]/[Email truy xuất]/"
            "[Tài liệu truy xuất]... của HỘI THOẠI bên dưới."
        )
    return (
        f"{src_block}\n\n{base}\n\n"
        "[Nhắc lại chế độ RAG: chấm GROUNDING — câu trả lời phải bám đúng NGUỒN, "
        "không khẳng định dữ kiện ngoài nguồn (UNGROUNDED), không bỏ sót dữ kiện nguồn "
        "mà câu hỏi cần (SOURCE_OMISSION). Số liệu/ngày/tên phải khớp tuyệt đối với nguồn.]"
    )


# ───────────────────────────────────────────────
# Judge API callers
# ───────────────────────────────────────────────
def _call_gemini(prompt: str, system: str = JUDGE_SYSTEM, file_path=None,
                 judge_model: str = JUDGE_MODEL) -> dict:
    if not _gemini_client:
        raise ValueError("Thiếu GEMINI_API_KEY trong .env để chạy Gemini judge.")
    contents = [prompt]
    if file_path:
        try:
            uploaded = _gemini_client.files.upload(file=str(file_path))
            contents = [uploaded, prompt]
        except Exception as e:
            # Fallback: không upload được thì đọc text nhúng vào prompt.
            print(f" [RAG upload file lỗi: {e} → fallback đọc text] ", end="", flush=True)
            try:
                txt = Path(file_path).read_text(encoding="utf-8")
                contents = [f'NGUỒN (đọc từ file):\n"""\n{txt}\n"""\n\n{prompt}']
            except Exception:
                pass
    resp = _gemini_client.models.generate_content(
        model=judge_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=JUDGE_TEMPERATURE,
            seed=JUDGE_SEED,
            response_mime_type="application/json",
            response_schema=JUDGE_SCHEMA,
        ),
    )
    return json.loads(resp.text)


def _call_deepseek(prompt: str, system: str = JUDGE_SYSTEM,
                   judge_model: str = JUDGE_MODEL) -> dict:
    if not _deepseek_client:
        raise ValueError("Thiếu DEEPSEEK_API_KEY trong .env để chạy DeepSeek judge.")

    kwargs = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": JUDGE_TEMPERATURE,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if DEEPSEEK_REASONING_EFFORT:
        kwargs["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
    if DEEPSEEK_THINKING_ENABLED:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    completion = _deepseek_client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content or "{}"
    return json.loads(content)


def _call_openrouter(prompt: str, system: str = JUDGE_SYSTEM,
                     judge_model: str = JUDGE_MODEL) -> dict:
    if not _openrouter_client:
        raise ValueError("Thiếu OPENROUTER_API_KEY trong .env để chạy OpenRouter judge.")

    completion = _openrouter_client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=JUDGE_TEMPERATURE,
        seed=JUDGE_SEED,
        response_format={"type": "json_object"},
        stream=False,
    )
    content = completion.choices[0].message.content or "{}"
    return json.loads(content)


# ───────────────────────────────────────────────
# Verdict post-processing
# ───────────────────────────────────────────────
def _strip_thinking(text: str) -> str:
    """Loại block <thinking>...</thinking> để chỉ chấm phần hiển thị cho user."""
    if "<thinking>" in text and "</thinking>" in text:
        parts = text.split("</thinking>")
        if len(parts) > 1:
            return parts[-1]
    return text


def _apply_must_not_contain_override(data: dict, output_text: str, case: dict) -> dict:
    """Safety net: nếu output chứa cụm cấm ngoài context refusal, cap overall ≤ 35."""
    must_not_contain = (case.get("expected", {}) or {}).get("must_not_contain", [])
    if not must_not_contain:
        return data

    clean_text = _strip_thinking(output_text)
    strict_language_check = (case.get("language") == "en")

    for forbidden in must_not_contain:
        if not _has_unsafe_assertion(clean_text, forbidden, strict=strict_language_check):
            continue

        data["overall"] = min(int(data.get("overall", 0)), 35)
        per_tc = data.get("per_tc") or {}
        violations = data.get("violations") or []

        if strict_language_check:
            tag = "POOR_LANGUAGE"
            affected = "TC-05"
            override_note = (
                f"[Override] Case yêu cầu 100% tiếng Anh nhưng chứa cụm Việt cấm '{forbidden}'."
            )
        else:
            tag = "HALLUCINATION"
            affected = "TC-01"
            override_note = (
                f"[Override] Khẳng định cụm cấm '{forbidden}' ngoài ngữ cảnh refusal — dấu hiệu bịa."
            )

        if affected in per_tc:
            entry = per_tc[affected]
            if isinstance(entry, dict):
                entry["score"] = min(int(entry.get("score", 0)), 35)
                entry["note"] = override_note
            else:
                per_tc[affected] = {"score": 35, "note": override_note}
        data["per_tc"] = per_tc

        if tag not in violations:
            violations.append(tag)
        data["violations"] = violations
        data["reasoning"] = override_note
        break

    return data


def _sanitize_verdict(data: dict, case: dict) -> dict:
    """Clamp scores về [0,100], dedup violations, filter per_tc theo criteria của case."""
    if data.get("violations"):
        seen = set()
        deduped = []
        for v in data["violations"]:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        data["violations"] = deduped

    overall = max(0, min(100, int(data.get("overall", 0))))
    data["overall"] = overall

    allowed = set(case.get("criteria", []) or [])
    raw_per_tc = data.get("per_tc") or {}
    cleaned_per_tc = {}
    for tc, v in raw_per_tc.items():
        if tc not in allowed:
            continue
        if isinstance(v, dict):
            try:
                score = max(0, min(100, int(v.get("score", 0))))
            except (ValueError, TypeError):
                continue
            note = str(v.get("note", "")).strip()
            cleaned_per_tc[tc] = {"score": score, "note": note}
        else:
            # Backward-compat: judge từng trả về int (schema cũ).
            try:
                cleaned_per_tc[tc] = {"score": max(0, min(100, int(v))), "note": ""}
            except (ValueError, TypeError):
                continue
    data["per_tc"] = cleaned_per_tc
    data["passed"] = overall >= PASS_THRESHOLD
    return data


# ───────────────────────────────────────────────
# Top-level judge entrypoint
# ───────────────────────────────────────────────
def judge(case, output_text, system_prompt="", retries=2, judge_model: str | None = None):
    judge_model = judge_model or JUDGE_MODEL
    is_openrouter = "/" in judge_model
    is_deepseek = judge_model.startswith("deepseek")
    # Bản RAG: case có data truy xuất (type rag_with_data) hoặc khai báo rag_source.
    is_rag = case.get("type") == "rag_with_data" or bool(case.get("rag_source"))

    file_path = None
    if is_rag:
        kind, src = _resolve_rag_source(case)
        source_text = src if kind == "text" else None
        if kind == "file":
            if is_deepseek or is_openrouter:
                # Non-Gemini providers here do not use Gemini Files API → đọc text nhúng vào prompt.
                try:
                    source_text = Path(src).read_text(encoding="utf-8")
                except Exception:
                    source_text = None
            else:
                file_path = src
        prompt = build_judge_prompt_rag(
            case, output_text, system_prompt, source_text, has_file=bool(file_path)
        )
        judge_system = JUDGE_SYSTEM_RAG
    else:
        prompt = build_judge_prompt(case, output_text, system_prompt)
        judge_system = JUDGE_SYSTEM

    last_err = None

    for attempt in range(retries + 1):
        try:
            _rate_limit_wait()
            data = (
                _call_openrouter(prompt, judge_system, judge_model) if is_openrouter
                else _call_deepseek(prompt, judge_system, judge_model) if is_deepseek
                else _call_gemini(prompt, judge_system, file_path, judge_model)
            )
            data = _apply_must_not_contain_override(data, output_text, case)
            return _sanitize_verdict(data, case)
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str:
                wait = 65.0
                print(f" [429: sleep {wait:.0f}s để reset RPM] ", end="", flush=True)
                _judge_call_times.clear()
            else:
                wait = 2 * (attempt + 1)
            time.sleep(wait)

    return {
        "passed": False,
        "overall": 0,
        "per_tc": {},
        "reasoning": f"Judge error: {last_err}",
        "violations": ["OTHER"],
    }
