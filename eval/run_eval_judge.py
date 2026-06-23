"""LLM-as-Judge evaluation runner — orchestrator.

Pipeline:
  1. Đọc bộ test theo --set: test_cases_chat.yaml hoặc test_cases_rag.yaml
  2. Gọi Modal model (SDK hoặc HTTP) → lấy response
  3. Gửi (câu hỏi + eval_notes + response) cho Gemini/Qwen làm judge
  4. Judge trả JSON → tổng hợp markdown report

Bộ test (config.TEST_SET, mặc định 'chat'):
  --set chat   → test_cases_chat.yaml  (case chat thuần, data inline)
  --set rag    → test_cases_rag.yaml   (case cần truy xuất: baseline refuse + biến thể .rag)
  Output/report gắn tên set: eval_outputs__<set>__<model>.json, eval_report_judge__<set>__<model>.md
  → chạy 2 lần (chat và rag) để có đủ 2 báo cáo.

Chạy:
  python eval/run_eval_judge.py                       # set chat, all, mode sdk, judge mặc định
  python eval/run_eval_judge.py --set rag             # set rag, all
  python eval/run_eval_judge.py generate --set rag    # chỉ stage generate cho set rag
  python eval/run_eval_judge.py judge                 # chỉ stage judge (cần outputs.json tương ứng)
  python eval/run_eval_judge.py judge sdk gemini-2.5-pro --set rag

Gen riêng bản CHƯA RAG vs bản CÓ RAG (--variant, trong set rag):
  python eval/run_eval_judge.py --set rag --variant base   # chỉ baseline (refuse khi thiếu data)
  python eval/run_eval_judge.py --set rag --variant rag    # chỉ bản có data truy xuất (.rag, PRO.*, FLW.v2)
  → output tách: eval_outputs__rag__<model>__base.json  vs  ...__rag.json

Test riêng từng case (--id, phân tách bằng dấu phẩy):
  python eval/run_eval_judge.py --set rag --id T01.v1.rag     # chạy đúng 1 case
  python eval/run_eval_judge.py --id EMO.v1,EMO.v2            # vài case (set chat mặc định)
  → ghi ra eval_outputs__<set>__<model>__subset.json + report __subset.md (không đè bản đầy đủ)
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from config import (
    CASE_IDS,
    GENERATE_WORKERS,
    JUDGE_MODELS,
    MULTI_JUDGE_REPORT_PATH,
    MODE,
    OUTPUTS_JSON_PATH,
    PASS_THRESHOLD,
    STAGE,
    TARGET_TEMPERATURE,
    TARGET_MODEL_NAME,
    TEST_CASES_PATH,
    TEST_SET,
    VARIANT,
    report_path_for_judge,
)
from judge import judge
from report import write_report
from target import call_target


# ───────────────────────────────────────────────
# Stage 1 — Generate responses
# ───────────────────────────────────────────────
_EVAL_DIR = TEST_CASES_PATH.parent

# Case `language: en`: model đôi khi drift sang tiếng Việt ở lượt đầu (prior tiếng
# Việt của system prompt áp đảo). Theo thiết kế, ta đánh giá ngôn ngữ theo LUỒNG
# 2 LƯỢT: sinh đáp án lượt 1 → tự chèn turn user yêu cầu viết lại bằng tiếng Anh →
# sinh đáp án lượt 2. Judge chấm TC-05 trên đáp án CUỐI: recover đúng tiếng Anh = PASS.
EN_RECOVERY_PROMPT = "Please write everything in English."


def _read_rag_source(src):
    """rag_source có thể là đường dẫn file (so với eval/) hoặc text nội tuyến."""
    p = _EVAL_DIR / str(src)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return str(src)


def _build_messages(case, system_prompt):
    messages = [dict(t) for t in case.get("turns", [])]
    # Nếu case khai báo rag_source (file/text), chèn tài liệu truy xuất vào
    # đầu turn user đầu tiên để target "đọc" được như ngữ cảnh RAG.
    src = case.get("rag_source")
    if src:
        doc = _read_rag_source(src)
        for t in messages:
            if t.get("role") == "user":
                t["content"] = f"[Tài liệu truy xuất]\n{doc}\n\n{t.get('content', '')}"
                break
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages


def _run_one(case, system_prompt):
    """Gọi target, trả về dict các metric.

    Case `language: en`: sinh thêm 1 lượt 'recovery' (chèn turn user
    EN_RECOVERY_PROMPT) và lấy đáp án lượt 2 làm `output` cần judge. Đáp án lượt 1
    lưu ở `turn1_output` để report/judge thấy được nguyên luồng. Latency/tokens cộng
    dồn cả 2 lượt.
    """
    messages = _build_messages(case, system_prompt)
    thinking_mode = case.get("thinking_mode", False) or case.get("thinking_compare", False)
    res, latency = call_target(messages, thinking_mode)
    result = {
        "output": res.get("text", ""),
        "latency": latency,
        "prompt_tokens": res.get("prompt_tokens", 0),
        "completion_tokens": res.get("completion_tokens", 0),
        # RAG live: nguồn model đã tự truy xuất (để judge đối chiếu grounding + xem recall).
        "retrieved": res.get("retrieved", []),
        "retrieved_context": res.get("retrieved_context", ""),
    }

    if case.get("language") == "en":
        convo2 = messages + [
            {"role": "assistant", "content": res.get("text", "")},
            {"role": "user", "content": EN_RECOVERY_PROMPT},
        ]
        res2, latency2 = call_target(convo2, thinking_mode)
        result["turn1_output"] = res.get("text", "")
        result["recovery_prompt"] = EN_RECOVERY_PROMPT
        result["output"] = res2.get("text", "")
        result["latency"] = latency + latency2
        result["prompt_tokens"] = res.get("prompt_tokens", 0) + res2.get("prompt_tokens", 0)
        result["completion_tokens"] = res.get("completion_tokens", 0) + res2.get("completion_tokens", 0)

    return result


def _augment_case_for_recovery(case, case_output):
    """Trả case có `turns` mở rộng để judge thấy nguyên luồng 2 lượt EN recovery.

    Chèn vào sau turns gốc: đáp án lượt 1 (assistant) + turn user yêu cầu tiếng Anh.
    `output` cần judge vẫn là đáp án lượt 2 (đã nằm trong case_output['output']).
    Case không phải EN recovery → trả nguyên case.
    """
    t1 = case_output.get("turn1_output")
    if case.get("language") != "en" or t1 is None:
        return case
    c = dict(case)
    c["turns"] = list(case.get("turns", [])) + [
        {"role": "assistant", "content": t1},
        {"role": "user", "content": case_output.get("recovery_prompt", EN_RECOVERY_PROMPT)},
    ]
    return c


def _inject_retrieved_source(case, case_output):
    """RAG live: dùng nội dung model đã TỰ truy xuất làm NGUỒN inline cho judge,
    để judge chấm grounding (UNGROUNDED/SOURCE_OMISSION) đúng như set 'rag'.

    Set `rag_source` = chuỗi text (KHÔNG phải path) → judge.py bật chế độ RAG và
    dùng nó làm source_text. Rỗng (model refuse / không có data) → giữ nguyên case
    (judge chấm hành vi refuse qua must_not_contain)."""
    ctx = case_output.get("retrieved_context")
    if not ctx:
        return case
    c = dict(case)
    c["rag_source"] = ctx
    return c


def _error_run(error):
    return {
        "output": "",
        "latency": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": str(error),
    }


def _generate_job(job, system_prompt):
    case_index, total_cases, case, run_index, n_runs = job
    cid = case.get("id")
    try:
        return case_index, total_cases, cid, run_index, n_runs, _run_one(case, system_prompt)
    except Exception as e:
        return case_index, total_cases, cid, run_index, n_runs, _error_run(e)


def _log_generate_result(done, total_jobs, cid, run_index, n_runs, run):
    run_suffix = f" run {run_index + 1}/{n_runs}" if n_runs > 1 else ""
    if run.get("error"):
        print(f"[{done}/{total_jobs}] {cid}{run_suffix} ERROR: {run['error']}")
    else:
        print(f"[{done}/{total_jobs}] {cid}{run_suffix} Done ({run['latency']:.1f}s)")


def generate_responses(test_cases, system_prompt):
    print(f"Giai doan GENERATE: Sinh cau tra loi cho {len(test_cases)} test cases...")
    print(
        f"  Target: Modal {MODE.upper()} | Model: {TARGET_MODEL_NAME} | "
        f"Workers: {GENERATE_WORKERS} | Temperature: {TARGET_TEMPERATURE}\n"
    )

    # `_meta` lưu tên model + mode để judge stage hiển thị trong report.
    outputs = {
        "_meta": {
            "model": TARGET_MODEL_NAME,
            "mode": MODE,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    case_specs = []
    jobs = []
    for i, case in enumerate(test_cases, 1):
        cid = case.get("id")
        n_runs = max(1, int(case.get("rerun", 1)))
        case_specs.append((case, cid, n_runs))
        for run_index in range(n_runs):
            jobs.append((i, len(test_cases), case, run_index, n_runs))

    results = {
        cid: [None] * n_runs
        for _, cid, n_runs in case_specs
    }
    total_jobs = len(jobs)
    done = 0

    if GENERATE_WORKERS == 1:
        for job in jobs:
            _, _, cid, run_index, n_runs, run = _generate_job(job, system_prompt)
            results[cid][run_index] = run
            done += 1
            _log_generate_result(done, total_jobs, cid, run_index, n_runs, run)
    else:
        with ThreadPoolExecutor(max_workers=GENERATE_WORKERS) as executor:
            futures = [
                executor.submit(_generate_job, job, system_prompt)
                for job in jobs
            ]
            for future in as_completed(futures):
                _, _, cid, run_index, n_runs, run = future.result()
                results[cid][run_index] = run
                done += 1
                _log_generate_result(done, total_jobs, cid, run_index, n_runs, run)

    for case, cid, n_runs in case_specs:
        runs = [
            run if run is not None else _error_run("Generate job did not return a result")
            for run in results[cid]
        ]
        if n_runs == 1:
            outputs[cid] = {"model": TARGET_MODEL_NAME, **runs[0]}
        elif all(run.get("error") for run in runs):
            outputs[cid] = {
                "model": TARGET_MODEL_NAME,
                "runs": runs,
                "error": "; ".join(run["error"] for run in runs),
            }
        else:
            # Multi-run case (phục vụ TC-09 Consistency).
            outputs[cid] = {"model": TARGET_MODEL_NAME, "runs": runs}

    with open(OUTPUTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Da luu cau tra loi vao {OUTPUTS_JSON_PATH}\n")


# ───────────────────────────────────────────────
# Stage 2 — Judge
# ───────────────────────────────────────────────
def _empty_result(case, reason):
    """Result skeleton dùng cho case skip (target error) hoặc judge fail."""
    return {
        "id": case.get("id"),
        "name": case.get("name", ""),
        "use_case": case.get("use_case"),
        "criteria": case.get("criteria", []),
        "passed": False,
        "overall": 0,
        "per_tc": {},
        "reasoning": reason,
        "violations": ["execution_error"],
        "judge_thinking": "",
        "latency": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "output": "",
        "turns": case.get("turns", []),
        "eval_notes": (case.get("expected", {}) or {}).get("eval_notes", ""),
    }


def _judge_safely(case, output_text, system_prompt, judge_model, run_label=""):
    try:
        v = judge(case, output_text, system_prompt, judge_model=judge_model)
        return v, None
    except Exception as e:
        v = {
            "passed": False, "overall": 0, "per_tc": {},
            "reasoning": f"Judge execution error{run_label}: {e}",
            "violations": ["OTHER"], "judge_thinking": "",
        }
        return v, e


def _judge_multi_run(case, case_output, system_prompt, judge_model):
    """Chấm từng run, lấy worst-case + apply TC-09 consistency penalty."""
    runs = case_output["runs"]
    sub_verdicts = []
    sub_errs = []
    for k, run in enumerate(runs):
        v, err = _judge_safely(case, run.get("output", ""), system_prompt, judge_model,
                                run_label=f" (run {k+1})")
        sub_verdicts.append(v)
        sub_errs.append(err)

    overalls = [int(v.get("overall", 0)) for v in sub_verdicts]
    min_overall = min(overalls)
    variance = max(overalls) - min_overall

    # Dùng run có điểm thấp nhất làm verdict chính.
    worst_idx = overalls.index(min_overall)
    verdict = dict(sub_verdicts[worst_idx])
    base_reason = verdict.get("reasoning", "")
    verdict["reasoning"] = (
        f"[Multi-run x{len(runs)} | overall={overalls}, variance={variance}] {base_reason}"
    )

    # TC-09 Consistency penalty: variance ≥ 20 trên scale 0-100 → trừ thêm.
    if "TC-09" in case.get("criteria", []) and variance >= 20:
        penalty = min(20, variance // 2)
        penalized = max(0, min_overall - penalty)
        verdict["overall"] = penalized
        verdict["passed"] = penalized >= PASS_THRESHOLD
        per_tc = dict(verdict.get("per_tc") or {})
        tc09_note = f"Multi-run variance={variance} → các lần chạy không nhất quán."
        if "TC-09" in per_tc and isinstance(per_tc["TC-09"], dict):
            per_tc["TC-09"] = {
                "score": min(int(per_tc["TC-09"].get("score", 0)), 40),
                "note": tc09_note,
            }
        else:
            per_tc["TC-09"] = {"score": 40, "note": tc09_note}
        verdict["per_tc"] = per_tc
        violations = list(verdict.get("violations") or [])
        if "OTHER" not in violations:
            violations.append("OTHER")
        verdict["violations"] = violations
        verdict["reasoning"] = (
            f"[TC-09 inconsistency penalty: variance={variance} ≥ 20 → -{penalty}] "
            + verdict["reasoning"]
        )

    # Aggregate latency/tokens + ghép output để debug.
    latency = sum(r.get("latency", 0.0) for r in runs) / len(runs)
    p_tokens = sum(r.get("prompt_tokens", 0) for r in runs)
    c_tokens = sum(r.get("completion_tokens", 0) for r in runs)
    output_text = "\n\n--- RUN SEPARATOR ---\n\n".join(
        f"[Run {k+1}/{len(runs)} | overall={overalls[k]}]\n{r.get('output','')}"
        for k, r in enumerate(runs)
    )
    judge_err = next((e for e in sub_errs if e), None)
    return verdict, latency, p_tokens, c_tokens, output_text, judge_err


def _judge_single_run(case, case_output, system_prompt, judge_model):
    output_text = case_output.get("output", "")
    latency = case_output.get("latency", 0.0)
    p_tokens = case_output.get("prompt_tokens", 0)
    c_tokens = case_output.get("completion_tokens", 0)
    verdict, err = _judge_safely(case, output_text, system_prompt, judge_model)
    return verdict, latency, p_tokens, c_tokens, output_text, err


def _log_case_result(r, judge_err):
    """Print 1 dòng status. Bọc try/except để encoding crash không phá pipeline."""
    try:
        if judge_err:
            print(f" ERROR: {judge_err}")
        elif r["passed"]:
            print(f" PASS (overall={r['overall']}/100)")
        else:
            print(f" FAIL (overall={r['overall']}/100) -- {r['reasoning'][:80]}")
    except UnicodeEncodeError:
        print(f" [overall={r['overall']}/100 — log unicode skipped]")


def _run_judge_stage_for_model(test_cases, system_prompt, outputs, meta, judge_model):
    print(f"Giai doan JUDGE: Cham diem bang {judge_model}...")
    results = []
    passed_count = 0
    total_latency = 0.0
    total_prompt = 0
    total_completion = 0
    total_score = 0

    for i, case in enumerate(test_cases, 1):
        cid = case.get("id")
        print(f"[{i}/{len(test_cases)}] Judge {judge_model} -> {cid}...", end="", flush=True)

        case_output = outputs.get(cid, {})

        if case_output.get("error"):
            print(f" SKIP due to target model error: {case_output['error']}")
            results.append(_empty_result(case, f"Original execution error: {case_output['error']}"))
            continue

        if "runs" in case_output:
            verdict, latency, p_tokens, c_tokens, output_text, judge_err = \
                _judge_multi_run(case, case_output, system_prompt, judge_model)
            judge_case = case
        else:
            # EN recovery: judge thấy nguyên luồng 2 lượt (turns mở rộng).
            judge_case = _augment_case_for_recovery(case, case_output)
            # RAG live: gắn nguồn đã truy xuất để judge chấm grounding.
            judge_case = _inject_retrieved_source(judge_case, case_output)
            verdict, latency, p_tokens, c_tokens, output_text, judge_err = \
                _judge_single_run(judge_case, case_output, system_prompt, judge_model)

        r = {
            "id": cid,
            "name": case.get("name", ""),
            "use_case": case.get("use_case"),
            "criteria": case.get("criteria", []),
            "passed": bool(verdict.get("passed")),
            "overall": int(verdict.get("overall", 0)),
            "per_tc": dict(verdict.get("per_tc") or {}),
            "reasoning": verdict.get("reasoning", ""),
            "violations": verdict.get("violations", []),
            "judge_thinking": verdict.get("judge_thinking", ""),
            "latency": latency,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "output": output_text,
            "turns": judge_case.get("turns", []),
            "eval_notes": (case.get("expected", {}) or {}).get("eval_notes", ""),
        }
        results.append(r)
        total_latency += latency
        total_prompt += p_tokens
        total_completion += c_tokens
        total_score += r["overall"]
        if not judge_err and r["passed"]:
            passed_count += 1
        _log_case_result(r, judge_err)

    report_path = report_path_for_judge(judge_model)
    write_report(test_cases, results, passed_count, total_latency, total_prompt,
                  total_completion, total_score, target_meta=meta,
                  judge_model=judge_model, report_path=report_path)
    return {
        "judge_model": judge_model,
        "report_path": str(report_path),
        "results": results,
        "passed_count": passed_count,
        "total_score": total_score,
    }


def _write_multi_judge_report(test_cases, judge_runs, target_meta):
    """Write a compact consensus report across all judge models."""
    by_judge = {
        run["judge_model"]: {r["id"]: r for r in run["results"]}
        for run in judge_runs
    }
    n = len(test_cases) or 1
    consensus_pass = 0
    avg_scores = []

    with open(MULTI_JUDGE_REPORT_PATH, "w", encoding="utf-8") as rf:
        rf.write("# Bao cao Multi-Judge Consensus\n\n")
        rf.write(f"* **Target model**: `{target_meta.get('model', TARGET_MODEL_NAME)}`\n")
        rf.write(f"* **Judges**: {', '.join(f'`{j}`' for j in by_judge)}\n")
        rf.write(f"* **Test cases**: {len(test_cases)}\n")
        rf.write(f"* **Rule**: consensus PASS khi da so judge PASS; Avg score la trung binh overall.\n\n")
        rf.write("| ID | Majority | Avg | Min-Max | Per judge |\n")
        rf.write("|---|---|---|---|---|\n")

        for case in test_cases:
            cid = case.get("id")
            rows = [by_judge[j].get(cid) for j in by_judge]
            rows = [r for r in rows if r]
            if not rows:
                continue
            scores = [int(r.get("overall", 0)) for r in rows]
            passes = sum(1 for r in rows if r.get("passed"))
            majority = passes > len(rows) / 2
            consensus_pass += int(majority)
            avg = sum(scores) / len(scores)
            avg_scores.append(avg)
            per_judge = "; ".join(
                f"{judge}={by_judge[judge][cid]['overall']}"
                f"({'P' if by_judge[judge][cid]['passed'] else 'F'})"
                for judge in by_judge
                if cid in by_judge[judge]
            )
            rf.write(
                f"| `{cid}` | {'PASS' if majority else 'FAIL'} ({passes}/{len(rows)}) | "
                f"{avg:.1f} | {min(scores)}-{max(scores)} | {per_judge} |\n"
            )

        avg_overall = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
        pass_pct = consensus_pass / n * 100
        rf.write("\n## Tong ket\n\n")
        rf.write(f"* **Consensus PASS**: {consensus_pass}/{n} ({pass_pct:.1f}%)\n")
        rf.write(f"* **Avg consensus score**: {avg_overall:.1f}/100\n")
        rf.write("\n## Report rieng tung judge\n\n")
        for run in judge_runs:
            rf.write(f"* `{run['judge_model']}`: `{run['report_path']}`\n")

    print(f"\nBao cao multi-judge: {MULTI_JUDGE_REPORT_PATH}")


def run_judge_stage(test_cases, system_prompt):
    if not OUTPUTS_JSON_PATH.exists():
        print(f"Loi: Khong tim thay tep {OUTPUTS_JSON_PATH}. Hay chay stage 'generate' truoc!")
        sys.exit(1)

    print(f"Doc cau tra loi da sinh tu {OUTPUTS_JSON_PATH}")
    with open(OUTPUTS_JSON_PATH, "r", encoding="utf-8") as f:
        outputs = json.load(f)

    meta = outputs.get("_meta") or {}
    if meta:
        print(f"  -> Output sinh bởi: {meta.get('model','?')} ({meta.get('mode','?')}) "
              f"luc {meta.get('generated_at','?')}")

    judge_runs = []
    for judge_model in JUDGE_MODELS:
        judge_runs.append(_run_judge_stage_for_model(
            test_cases, system_prompt, outputs, meta, judge_model
        ))

    if len(judge_runs) > 1:
        _write_multi_judge_report(test_cases, judge_runs, meta)


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
def main():
    if not TEST_CASES_PATH.exists():
        print(f"Khong tim thay {TEST_CASES_PATH}")
        sys.exit(1)

    print(f"Doc test cases tu {TEST_CASES_PATH}")
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    test_cases = data.get("test_cases", [])
    system_prompt = data.get("system_prompt", "")

    # --variant: gen/judge riêng bản chưa RAG (base) hoặc bản có RAG (rag_with_data).
    if VARIANT != "all":
        before = len(test_cases)
        if VARIANT == "rag":
            test_cases = [c for c in test_cases if c.get("type") == "rag_with_data"]
        else:  # base
            test_cases = [c for c in test_cases if c.get("type") != "rag_with_data"]
        print(f"   [--variant {VARIANT}] Loc {len(test_cases)}/{before} case "
              f"({'co RAG' if VARIANT == 'rag' else 'chua RAG'}).")
        if not test_cases:
            print(f"Khong co case '{VARIANT}' trong set '{TEST_SET}'. Dung.")
            sys.exit(1)

    # --id: chỉ chạy các case được chỉ định (test riêng từng cái).
    if CASE_IDS:
        wanted = set(CASE_IDS)
        available = {c.get("id") for c in test_cases}
        test_cases = [c for c in test_cases if c.get("id") in wanted]
        missing = wanted - available
        if missing:
            print(f"⚠️ Khong thay ID trong set '{TEST_SET}': {sorted(missing)} "
                  f"(co the nam o set khac — thu doi --set).")
        if not test_cases:
            print("Khong co case nao khop --id. Dung.")
            sys.exit(1)
        print(f"   [--id] Loc {len(test_cases)} case: {[c.get('id') for c in test_cases]}")

    judge_label = ", ".join(JUDGE_MODELS)
    print(f"SET: {TEST_SET.upper()} | STAGE: {STAGE.upper()} | Target: {MODE.upper()} | "
          f"Judge: {judge_label} | Total cases: {len(test_cases)}\n")

    if STAGE == "generate":
        generate_responses(test_cases, system_prompt)
    elif STAGE == "judge":
        run_judge_stage(test_cases, system_prompt)
    else:
        generate_responses(test_cases, system_prompt)
        run_judge_stage(test_cases, system_prompt)


if __name__ == "__main__":
    main()
