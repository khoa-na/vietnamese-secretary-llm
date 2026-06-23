"""Modal app rieng de chay Gemma 4 12B GGUF bang llama.cpp.

Deploy:
    modal deploy modal_gguf_app.py

Eval bang Modal SDK:
    $env:MODAL_APP_NAME="test-llm-chatbot-thuky-gguf"
    $env:MODEL_NAME="unsloth/gemma-4-12b-it-GGUF"
    .venv\\Scripts\\python eval/run_eval_judge.py generate --set chat

App nay giu interface `LLMServer.generate(...)` giong `modal_app.py` de runner
co the goi lai qua Modal SDK. No khong thay the vLLM app hien tai.
"""

import json
import os
import re
from datetime import date
from pathlib import Path

import modal
from dotenv import dotenv_values, load_dotenv

load_dotenv()
env_path = Path(__file__).parent / ".env"
env_config = {
    **dotenv_values(env_path),
    **os.environ,
}

MODEL_NAME = env_config.get("GGUF_REPO_ID", "unsloth/gemma-4-12b-it-GGUF")
GGUF_FILENAME = env_config.get("GGUF_FILENAME", "gemma-4-12b-it-UD-Q4_K_XL.gguf")
MAX_MODEL_LEN = int(env_config.get("MAX_MODEL_LEN", "16384"))
GPU_TYPE = env_config.get("MODAL_GGUF_GPU", env_config.get("MODAL_GPU", "L40S"))
N_GPU_LAYERS = int(env_config.get("GGUF_N_GPU_LAYERS", "-1"))
N_BATCH = int(env_config.get("GGUF_N_BATCH", "512"))
N_THREADS = int(env_config.get("GGUF_N_THREADS", "8"))
VERBOSE = env_config.get("GGUF_VERBOSE", "false").lower() == "true"

USE_TOOLS_DEFAULT = env_config.get("USE_TOOLS", "true").lower() == "true"
PYTHON_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Run deterministic Python code for arithmetic, dates, counting, "
            "statistics, and exact transformations. The code must print the answer. "
            "Available stdlib: datetime, math, calendar, statistics, collections, re."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code that prints the final result.",
                }
            },
            "required": ["code"],
        },
    },
}
TOOL_CALL_WRAPPER_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\>\s*call:([A-Za-z0-9_]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)
GEMMA_TOOL_ARG_RE = re.compile(
    r"([A-Za-z0-9_]+):<\|\"\|>(.*?)<\|\"\|>",
    re.DOTALL,
)
JSON_INNER_RE = re.compile(r"^\s*(\{.*\})\s*$", re.DOTALL)
XML_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)
MAX_TOOL_ITERS = int(env_config.get("MAX_TOOL_ITERS", "4"))
SANDBOX_EXEC_TIMEOUT = int(env_config.get("SANDBOX_EXEC_TIMEOUT", "8"))
SANDBOX_LIFETIME = int(env_config.get("SANDBOX_LIFETIME", "1800"))

USE_RETRIEVAL_DEFAULT = env_config.get("USE_RETRIEVAL", "true").lower() == "true"
EMBED_MODEL = env_config.get("EMBED_MODEL", "BAAI/bge-m3")
RETRIEVAL_TOP_K = int(env_config.get("RAG_TOP_K", "4"))
RAG_MIN_SCORE = env_config.get("RAG_MIN_SCORE", "0.56")
RAG_ROUTE_MIN_SCORE = env_config.get("RAG_ROUTE_MIN_SCORE", "0.59")
RAG_REFERENCE_DATE = env_config.get("RAG_REFERENCE_DATE", "2026-05-27")
RAG_CORPUS_DIR = "/root/rag_corpus"

image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.21.0-cu129-ubuntu2404",
        add_python=None,
        setup_dockerfile_commands=["ENTRYPOINT []"],
    )
    .run_commands("ln -sf $(which python3) /usr/local/bin/python")
    .apt_install("build-essential", "cmake", "ninja-build", "git", "curl")
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --no-cache-dir 'huggingface_hub[hf_transfer]>=0.26' "
        "'sentence-transformers>=3.0' 'python-dotenv>=1.0'",
        # Build llama-cpp-python with CUDA support inside the image.
        "CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 "
        "python -m pip install --no-cache-dir --verbose 'llama-cpp-python>=0.3.16'",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "RAG_CORPUS_DIR": RAG_CORPUS_DIR,
            "RAG_REFERENCE_DATE": RAG_REFERENCE_DATE,
            "RAG_MIN_SCORE": RAG_MIN_SCORE,
            "RAG_ROUTE_MIN_SCORE": RAG_ROUTE_MIN_SCORE,
        }
    )
    .add_local_python_source("retrieval")
    .add_local_dir(str(Path(__file__).parent / "rag_corpus"), remote_path=RAG_CORPUS_DIR)
)

sandbox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("python-dateutil")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
llamacpp_cache = modal.Volume.from_name("llamacpp-cache", create_if_missing=True)

app = modal.App("test-llm-chatbot-thuky-gguf")


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/llama.cpp": llamacpp_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=5,
    timeout=900,
    min_containers=0,
)
class LLMServer:
    @modal.enter()
    def load_model(self):
        from llama_cpp import Llama

        hf_token = env_config.get("HF_TOKEN") or env_config.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            print(f"[gguf] HF_TOKEN detected: {hf_token[:6]}...{hf_token[-4:]}", flush=True)
        else:
            print("[gguf] No HF_TOKEN; only public models are available.", flush=True)

        print(
            f"[gguf] Loading {MODEL_NAME}:{GGUF_FILENAME} on {GPU_TYPE} "
            f"ctx={MAX_MODEL_LEN} n_gpu_layers={N_GPU_LAYERS}",
            flush=True,
        )
        self.llm = Llama.from_pretrained(
            repo_id=MODEL_NAME,
            filename=GGUF_FILENAME,
            n_ctx=MAX_MODEL_LEN,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=N_BATCH,
            n_threads=N_THREADS,
            verbose=VERBOSE,
        )
        print("[gguf] Model ready.", flush=True)

        self.retriever = None
        if USE_RETRIEVAL_DEFAULT:
            try:
                import retrieval
                from sentence_transformers import SentenceTransformer

                try:
                    ref_date = date.fromisoformat(RAG_REFERENCE_DATE)
                except ValueError:
                    ref_date = None
                print(f"[gguf] [rag] Loading embedder {EMBED_MODEL} on cuda ...", flush=True)
                embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
                self.retriever = retrieval.Retriever(
                    corpus_dir=RAG_CORPUS_DIR,
                    reference_date=ref_date,
                )
                self.retriever.build_index(embedder)
                print(f"[gguf] [rag] Retriever ready: {len(self.retriever.chunks)} chunks.", flush=True)
            except Exception as e:
                print(f"[gguf] [rag] Retriever init FAILED ({e}); running without retrieval.", flush=True)
                self.retriever = None
        self._sandbox = None

    @modal.exit()
    def cleanup(self):
        sb = getattr(self, "_sandbox", None)
        if sb is not None:
            try:
                sb.terminate()
                print("[gguf] [sandbox] terminated", flush=True)
            except Exception as e:
                print(f"[gguf] [sandbox] cleanup error: {e}", flush=True)

    def _get_sandbox(self):
        sb = getattr(self, "_sandbox", None)
        if sb is not None:
            try:
                if sb.returncode is None:
                    return sb
            except Exception:
                pass
        from modal import Sandbox

        sb = Sandbox.create(
            "sleep",
            "infinity",
            image=sandbox_image,
            app=app,
            timeout=SANDBOX_LIFETIME,
            block_network=True,
        )
        self._sandbox = sb
        print("[gguf] [sandbox] created", flush=True)
        return sb

    def _exec_in_sandbox(self, code: str) -> str:
        try:
            sb = self._get_sandbox()
            proc = sb.exec("python3", "-c", code, timeout=SANDBOX_EXEC_TIMEOUT)
            stdout = proc.stdout.read() or ""
            stderr = proc.stderr.read() or ""
            try:
                proc.wait()
            except Exception:
                pass
        except Exception as e:
            return f"[sandbox_error] {e}"
        out = stdout.rstrip()
        if stderr.strip():
            out = (out + "\n[stderr]\n" + stderr.rstrip()).strip()
        return out or "(no output)"

    @staticmethod
    def _parse_text_tool_calls(text: str):
        calls = []
        for match in GEMMA_TOOL_CALL_RE.finditer(text or ""):
            args = {
                arg_match.group(1).strip(): arg_match.group(2).strip()
                for arg_match in GEMMA_TOOL_ARG_RE.finditer(match.group(2))
            }
            calls.append({"name": match.group(1).strip(), "arguments": args})
        for match in TOOL_CALL_WRAPPER_RE.finditer(text or ""):
            inner = match.group(1).strip()
            json_match = JSON_INNER_RE.match(inner)
            if json_match:
                try:
                    calls.append(json.loads(json_match.group(1)))
                    continue
                except json.JSONDecodeError:
                    pass
            for fn_match in XML_FUNCTION_RE.finditer(inner):
                args = {
                    param_match.group(1).strip(): param_match.group(2).strip()
                    for param_match in XML_PARAM_RE.finditer(fn_match.group(2))
                }
                calls.append({"name": fn_match.group(1).strip(), "arguments": args})
        residual = GEMMA_TOOL_CALL_RE.sub("", text or "")
        residual = TOOL_CALL_WRAPPER_RE.sub("", residual).strip()
        return calls, residual

    @staticmethod
    def _extract_tool_calls(message: dict):
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({
                "id": call.get("id", f"call_{len(calls)}"),
                "name": fn.get("name", ""),
                "arguments": args,
            })
        text_calls, residual = LLMServer._parse_text_tool_calls(message.get("content", "") or "")
        for call in text_calls:
            calls.append({
                "id": f"call_{len(calls)}",
                "name": call.get("name", ""),
                "arguments": call.get("arguments", {}),
            })
        return calls, residual

    @staticmethod
    def _last_user_query(messages) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "") or ""
        return ""

    def _augment_with_retrieval(self, messages, enable_retrieval):
        convo = list(messages)
        if not (enable_retrieval and self.retriever is not None):
            return convo, [], ""
        query = self._last_user_query(messages)
        if not query:
            return convo, [], ""
        try:
            res = self.retriever.retrieve(query, top_k=RETRIEVAL_TOP_K)
        except Exception as e:
            print(f"[gguf] [rag] retrieve error: {e}", flush=True)
            return convo, [], ""
        if res.is_empty:
            return convo, [], ""
        context_text = "\n\n".join(res.blocks)
        aug = (
            f"[Du lieu truy xuat tu dong - hom nay {RAG_REFERENCE_DATE}]\n"
            + context_text
            + "\n\n(Chi dung du lieu tren + hoi thoai de tra loi; khong bia ngoai phan nay.)\n\n"
        )
        for i in range(len(convo) - 1, -1, -1):
            if convo[i].get("role") == "user":
                convo[i] = {**convo[i], "content": aug + (convo[i].get("content", "") or "")}
                break
        return convo, res.sources, context_text

    @modal.method()
    def retrieve_only(self, query: str, top_k: int | None = None) -> dict:
        if getattr(self, "retriever", None) is None:
            return {"blocks": [], "sources": [], "error": "retriever_unavailable"}
        res = self.retriever.retrieve(query, top_k=top_k or RETRIEVAL_TOP_K)
        return {
            "blocks": res.blocks,
            "sources": res.sources,
            "route_scores": self.retriever.route_scores(query),
        }

    @modal.method()
    def generate(
        self,
        messages: list | None = None,
        prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        thinking_mode: bool = False,
        use_tools: bool | None = None,
        use_retrieval: bool | None = None,
    ) -> dict:
        del thinking_mode

        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]
        if not messages:
            return {"error": "Missing messages or prompt."}

        enable_retrieval = USE_RETRIEVAL_DEFAULT if use_retrieval is None else bool(use_retrieval)
        convo, retrieved_sources, retrieved_context = self._augment_with_retrieval(
            messages,
            enable_retrieval,
        )

        enable_tools = USE_TOOLS_DEFAULT if use_tools is None else bool(use_tools)
        tools = [PYTHON_EXEC_TOOL] if enable_tools else None
        total_prompt_tokens = 0
        total_completion_tokens = 0
        tool_log = []
        finish_reason = ""
        final_text = ""
        force_final_answer = False

        for iteration in range(MAX_TOOL_ITERS + 1):
            kwargs = {
                "messages": convo,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if tools and not force_final_answer:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            response = self.llm.create_chat_completion(**kwargs)
            choice = response["choices"][0]
            usage = response.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            finish_reason = choice.get("finish_reason", "") or finish_reason
            message = choice.get("message", {}) or {}
            final_text = message.get("content", "") or ""

            if not tools:
                break

            calls, residual = self._extract_tool_calls(message)
            if force_final_answer or not calls or iteration == MAX_TOOL_ITERS:
                final_text = final_text or residual
                break

            assistant_msg = {
                "role": "assistant",
                "content": message.get("content", "") or "",
            }
            if message.get("tool_calls"):
                assistant_msg["tool_calls"] = message["tool_calls"]
            convo.append(assistant_msg)

            for i, call in enumerate(calls):
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                name = call.get("name", "")
                if name == "python_exec":
                    code = args.get("code", "") if isinstance(args, dict) else ""
                    result = self._exec_in_sandbox(code) if code else "[error] empty code"
                else:
                    code = ""
                    result = f"[error] unknown tool: {name}"
                tool_log.append({
                    "iter": iteration,
                    "name": name,
                    "code": code[:500],
                    "result": result[:500],
                })
                convo.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", f"call_{iteration}_{i}"),
                    "content": result[:2000],
                })
                convo.append({
                    "role": "user",
                    "content": (
                        "Ket qua tu python_exec:\n"
                        f"{result[:2000]}\n\n"
                        "Hay dung ket qua tren de tra loi cuoi cung cho nguoi dung "
                        "bang tieng Viet, ngan gon, ro rang. Khong goi tool nua."
                    ),
                })
            force_final_answer = True

        return {
            "text": final_text,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "finish_reason": finish_reason,
            "tool_calls": tool_log,
            "retrieved": retrieved_sources,
            "retrieved_context": retrieved_context,
        }
