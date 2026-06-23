"""Cấu hình + CLI args cho eval pipeline.

Một chỗ duy nhất chứa: env loading, parse args, constants, encoding setup.
Các module khác chỉ `from config import ...`.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# Windows console mặc định cp1252 → crash khi print Unicode tiếng Việt. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

load_dotenv()
_ENV_PATH = Path(__file__).parent.parent / ".env"
env_config = {
    **dotenv_values(_ENV_PATH),
    **os.environ,
}


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="run_eval_judge",
        description="LLM-as-Judge evaluation runner (Modal target + Gemini/Qwen judge).",
    )
    p.add_argument(
        "stage", nargs="?", default="all", choices=["generate", "judge", "all"],
        help="Stage: generate (chỉ sinh response), judge (chỉ chấm), all (mặc định).",
    )
    p.add_argument(
        "mode", nargs="?", default=None, choices=["sdk", "http"],
        help="Modal transport: sdk (default) hoặc http. Override EVAL_MODE.",
    )
    p.add_argument(
        "judge_model", nargs="?", default=None,
        help="Tên judge model. Override JUDGE_MODEL.",
    )
    p.add_argument(
        "--judge-models", dest="judge_models", default=None,
        help=(
            "Danh sách judge model phân tách bằng dấu phẩy. Override JUDGE_MODELS/JUDGE_MODEL. "
            "Vd: --judge-models gemini-2.5-flash,deepseek-v4-flash,qwen/qwen3.6-flash"
        ),
    )
    p.add_argument(
        "--set", dest="test_set", default=None, choices=["chat", "rag", "rag_live"],
        help="Bộ test: 'chat' (test_cases_chat.yaml), 'rag' (test_cases_rag.yaml — data dán sẵn), "
             "hoặc 'rag_live' (test_cases_rag_live.yaml — retrieval THẬT server-side). "
             "Override EVAL_SET (mặc định 'chat').",
    )
    p.add_argument(
        "--id", dest="case_ids", default=None,
        help="Chỉ chạy các case có ID này (phân tách bằng dấu phẩy) — để test riêng từng cái. "
             "Vd: --id T01.v1.rag  hoặc  --id EMO.v1,EMO.v2. Output ghi ra file *__subset.* riêng.",
    )
    p.add_argument(
        "--variant", dest="variant", default="all", choices=["base", "rag", "all"],
        help="Lọc trong bộ test theo type: 'base' = bản CHƯA có RAG (refuse baseline), "
             "'rag' = bản CÓ RAG (type rag_with_data, data đã truy xuất), 'all' = cả hai (mặc định). "
             "Output tách tên file theo variant để không đè nhau.",
    )
    # Hỗ trợ cú pháp cũ: thứ tự args có thể là bất kỳ — gom lại rồi reorder.
    # Vd `python run_eval_judge.py judge sdk gemini-2.5-flash` vẫn parse đúng.
    return p.parse_args(argv)


_args = _parse_args(sys.argv[1:])

STAGE: str = _args.stage
MODE: str = _args.mode or env_config.get("EVAL_MODE", "sdk")
if MODE not in ("sdk", "http"):
    print(f"⚠️ EVAL_MODE='{MODE}' không hỗ trợ. Fallback về 'sdk' (Modal SDK).")
    MODE = "sdk"

def _split_csv(raw: str | None) -> list[str]:
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


_judge_models_raw = _args.judge_models or env_config.get("JUDGE_MODELS")
JUDGE_MODELS: list[str] = (
    _split_csv(_judge_models_raw)
    or [_args.judge_model or env_config.get("JUDGE_MODEL", "gemini-3.1-flash-lite")]
)
JUDGE_MODEL: str = JUDGE_MODELS[0]

# Bộ test: chat (chỉ cần hội thoại) hoặc rag (cần data truy xuất). Mỗi bộ 1 file riêng.
TEST_SET: str = _args.test_set or env_config.get("EVAL_SET", "chat")
if TEST_SET not in ("chat", "rag", "rag_live"):
    print(f"⚠️ EVAL_SET='{TEST_SET}' không hỗ trợ. Fallback về 'chat'.")
    TEST_SET = "chat"

MAX_TOKENS: int = int(env_config.get("EVAL_MAX_TOKENS", "512"))
TARGET_TEMPERATURE: float = float(env_config.get("EVAL_TEMPERATURE", "0.2"))
# Bật python_exec tool cho target. None = theo USE_TOOLS_DEFAULT của Modal server.
_use_tools_raw = env_config.get("USE_TOOLS")
USE_TOOLS: bool | None = (
    None if _use_tools_raw is None else _use_tools_raw.strip().lower() == "true"
)
# Bật retrieval THẬT server-side. Mặc định: chỉ bật cho set 'rag_live' (các set khác
# data đã inline/self-contained nên KHÔNG retrieve để tránh chèn nhiễu). Override qua env.
_use_retrieval_raw = env_config.get("USE_RETRIEVAL")
USE_RETRIEVAL: bool = (
    (TEST_SET == "rag_live") if _use_retrieval_raw is None
    else _use_retrieval_raw.strip().lower() == "true"
)
GENERATE_WORKERS: int = max(1, int(env_config.get("EVAL_GENERATE_WORKERS", "1")))
JUDGE_TEMPERATURE: float = float(env_config.get("JUDGE_TEMPERATURE", "0.0"))
JUDGE_SEED: int = int(env_config.get("JUDGE_SEED", "42"))
JUDGE_RPM: int = int(env_config.get("JUDGE_RPM", "15"))

# Đọc cùng key MODEL_NAME như modal_app.py để đồng bộ.
TARGET_MODEL_NAME: str = env_config.get("MODEL_NAME", "Qwen/Qwen3.5-9B")

# Lọc theo case ID (chạy riêng 1 vài case). None = chạy cả bộ.
CASE_IDS: list[str] | None = (
    [s.strip() for s in _args.case_ids.split(",") if s.strip()]
    if _args.case_ids else None
)

# Lọc theo variant: base (chưa RAG) | rag (có RAG) | all.
VARIANT: str = _args.variant

# Slug an toàn cho filename: "Qwen/Qwen3.5-9B-Instruct" → "Qwen_Qwen3.5-9B-Instruct"
def slugify_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(",", "_")


_MODEL_SLUG = slugify_name(TARGET_MODEL_NAME)

# Suffix output để các lần chạy lọc khác nhau KHÔNG đè kết quả của nhau.
_SUFFIX = (f"__{VARIANT}" if VARIANT != "all" else "") + ("__subset" if CASE_IDS else "")

_EVAL_DIR = Path(__file__).parent

# Output gom vào folder riêng: results/generate (model gen) + results/judge (judge report).
RESULTS_DIR: Path = _EVAL_DIR / "results"
GEN_DIR: Path = RESULTS_DIR / "generate"
JUDGE_DIR: Path = RESULTS_DIR / "judge"
GEN_DIR.mkdir(parents=True, exist_ok=True)
JUDGE_DIR.mkdir(parents=True, exist_ok=True)

# Tên file gắn slug model + bộ test để dễ phân biệt (Qwen vs Llama, chat vs rag).
TEST_CASES_PATH: Path = _EVAL_DIR / f"test_cases_{TEST_SET}.yaml"
OUTPUTS_JSON_PATH: Path = GEN_DIR / f"eval_outputs__{TEST_SET}__{_MODEL_SLUG}{_SUFFIX}.json"
REPORT_PATH: Path = JUDGE_DIR / f"eval_report_judge__{TEST_SET}__{_MODEL_SLUG}{_SUFFIX}.md"


def report_path_for_judge(judge_model: str) -> Path:
    """Per-judge report path. Single-judge mode keeps the legacy filename."""
    if len(JUDGE_MODELS) == 1:
        return REPORT_PATH
    return JUDGE_DIR / (
        f"eval_report_judge__{TEST_SET}__{_MODEL_SLUG}__judge_{slugify_name(judge_model)}"
        f"{_SUFFIX}.md"
    )


MULTI_JUDGE_REPORT_PATH: Path = JUDGE_DIR / (
    f"eval_report_judge__{TEST_SET}__{_MODEL_SLUG}__multi_judge{_SUFFIX}.md"
)

GEMINI_API_KEY: str | None = env_config.get("GEMINI_API_KEY") or env_config.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY: str | None = env_config.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL: str = env_config.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_REASONING_EFFORT: str = env_config.get("DEEPSEEK_REASONING_EFFORT", "high")
DEEPSEEK_THINKING_ENABLED: bool = (
    env_config.get("DEEPSEEK_THINKING_ENABLED", "true").strip().lower() == "true"
)
OPENROUTER_API_KEY: str | None = env_config.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = env_config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL: str = env_config.get("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME: str = env_config.get("OPENROUTER_APP_NAME", "vietnamese-secretary-llm")
MODAL_APP_NAME: str = env_config.get("MODAL_APP_NAME", "test-llm-chatbot-thuky")
MODAL_ENDPOINT_URL: str = env_config.get("MODAL_ENDPOINT_URL", "")

# Ngưỡng pass / production-ready trên thang 0-100.
PASS_THRESHOLD: int = 60
PRODUCTION_THRESHOLD: int = 75
