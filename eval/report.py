"""Markdown report writer cho LLM-as-Judge eval pipeline."""
import time

from config import (
    JUDGE_MODEL,
    MODE,
    PASS_THRESHOLD,
    PRODUCTION_THRESHOLD,
    REPORT_PATH,
    TARGET_MODEL_NAME,
)
from criteria import CRITERIA_DEFINITIONS


def _compute_per_tc_scores(results):
    """Aggregate điểm 0-100 trung bình mỗi TC.

    Ưu tiên dùng `per_tc` breakdown của từng case; fallback `overall` nếu thiếu.
    Trả về dict {tc: {"avg": float, "count": int, "pass_rate": float}}.
    """
    bucket = {}
    for r in results:
        per_tc = r.get("per_tc") or {}
        for tc in r.get("criteria", []) or []:
            entry = per_tc.get(tc)
            if isinstance(entry, dict) and "score" in entry:
                v = int(entry["score"])
            elif isinstance(entry, int):
                v = entry
            else:
                v = int(r.get("overall", 0) or 0)
            bucket.setdefault(tc, []).append(v)
    summary = {}
    for tc in sorted(CRITERIA_DEFINITIONS.keys()):
        scores = bucket.get(tc, [])
        if scores:
            passes = sum(1 for s in scores if s >= PASS_THRESHOLD)
            summary[tc] = {
                "avg": sum(scores) / len(scores),
                "count": len(scores),
                "pass_rate": passes / len(scores) * 100,
            }
    return summary


def _per_tc_short(per_tc):
    """Hiển thị 1 dòng 'TC-02=80, TC-06=60' (chỉ score, cho bảng tổng)."""
    parts = []
    for tc, v in (per_tc or {}).items():
        if isinstance(v, dict):
            parts.append(f"{tc}={v.get('score', '?')}")
        else:
            parts.append(f"{tc}={v}")
    return ", ".join(parts) or "-"


def _print_console_summary(n, passed_count, production_count, avg_overall, avg_latency,
                            total_prompt, total_completion, target_meta, per_tc,
                            judge_model=None):
    pass_pct = passed_count / n * 100
    production_pct = production_count / n * 100
    print("\n" + "=" * 60)
    print("KET QUA LLM-AS-JUDGE (Thang 0-100)")
    print("=" * 60)
    print(f"Tong test cases    : {n}")
    print(f"PASS (>= {PASS_THRESHOLD})       : {passed_count} ({pass_pct:.1f}%)")
    print(f"Production (>= {PRODUCTION_THRESHOLD}): {production_count} ({production_pct:.1f}%)")
    print(f"FAIL               : {n - passed_count}")
    print(f"Overall TB         : {avg_overall:.1f}/100")
    print(f"Latency TB         : {avg_latency:.2f}s")
    print(f"Prompt tokens      : {total_prompt}")
    print(f"Completion tokens  : {total_completion}")
    print(f"Target model       : {target_meta.get('model', TARGET_MODEL_NAME)} "
          f"(Modal {target_meta.get('mode', MODE).upper()})")
    print(f"Judge model        : {judge_model or JUDGE_MODEL}")
    print("-" * 60)
    print("Diem TB theo tieu chi (xlsx v3, thang 0-100):")
    for tc, s in per_tc.items():
        print(f"  {tc}: {s['avg']:5.1f}/100  pass={s['pass_rate']:5.1f}%  (n={s['count']})")
    print("=" * 60)


def _write_header(rf, n, passed_count, production_count, avg_overall, avg_latency,
                   total_prompt, total_completion, target_meta, judge_model=None):
    pass_pct = passed_count / n * 100
    production_pct = production_count / n * 100
    target_model = target_meta.get("model", TARGET_MODEL_NAME)
    target_mode_str = target_meta.get("mode", MODE).upper()
    gen_at = target_meta.get("generated_at", "?")

    rf.write("# Bao cao Danh gia LLM-as-Judge\n\n")
    rf.write(f"* **Thoi gian**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    rf.write(f"* **Target model**: `{target_model}` (Modal {target_mode_str}, generated {gen_at})\n")
    rf.write(f"* **Judge model**: `{judge_model or JUDGE_MODEL}`\n")
    rf.write(f"* **Nguon tieu chi**: `Chatbot_ThuKy_UseCases_EvalCriteria_v3.xlsx` (sheet 'Tieu chi danh gia')\n")
    rf.write(f"* **Thang diem**: 0-100 (PASS >= {PASS_THRESHOLD}, Production-ready >= {PRODUCTION_THRESHOLD})\n")
    rf.write(f"* **Test cases**: {n}\n")
    rf.write(f"* **PASS**: **{passed_count}/{n} ({pass_pct:.1f}%)**\n")
    rf.write(f"* **Production-ready**: **{production_count}/{n} ({production_pct:.1f}%)**\n")
    rf.write(f"* **Overall trung binh**: **{avg_overall:.1f}/100**\n")
    rf.write(f"* **Latency TB**: {avg_latency:.2f}s\n")
    rf.write(f"* **Tokens**: In {total_prompt} | Out {total_completion}\n\n")


def _write_per_tc_table(rf, per_tc):
    rf.write("## Diem trung binh theo tung tieu chi (xlsx v3)\n\n")
    rf.write("> Lay trung binh cua `per_tc` breakdown qua tat ca case co khai bao TC do trong `criteria`. "
             "Xlsx khong khai bao trong so nen khong tinh tong co weight.\n\n")
    rf.write("| TC | Ten tieu chi | So case | Diem TB | Pass rate |\n")
    rf.write("|---|---|---|---|---|\n")
    for tc, s in per_tc.items():
        short = CRITERIA_DEFINITIONS[tc].split(":")[0]
        rf.write(
            f"| `{tc}` | {short} | {s['count']} | {s['avg']:.1f}/100 | "
            f"{s['pass_rate']:.1f}% |\n"
        )
    rf.write("\n")


def _row_status(r):
    if r["overall"] >= PRODUCTION_THRESHOLD:
        return "PROD"
    return "PASS" if r["passed"] else "FAIL"


def _write_summary_table(rf, results):
    rf.write("## Bang tong hop\n\n")
    rf.write("| ID | Use Case | Test | Trang thai | Overall | Per-TC | Latency | Tom tat |\n")
    rf.write("|---|---|---|---|---|---|---|---|\n")
    for r in results:
        reason = r["reasoning"].replace("\n", " ").replace("|", "\\|")[:120]
        rf.write(
            f"| `{r['id']}` | `{r['use_case']}` | {r['name']} | {_row_status(r)} | "
            f"{r['overall']}/100 | {_per_tc_short(r.get('per_tc'))} | "
            f"{r['latency']:.2f}s | {reason} |\n"
        )


def _write_per_tc_block(rf, per_tc, score_suffix=""):
    """In per_tc thành bullet list. `score_suffix` vd '/100' hoặc ''."""
    for tc, v in (per_tc or {}).items():
        if isinstance(v, dict):
            rf.write(f"  * `{tc}`: **{v.get('score', '?')}{score_suffix}** — {v.get('note', '')}\n")
        else:
            rf.write(f"  * `{tc}`: **{v}{score_suffix}**\n")


def _write_fail_details(rf, results):
    rf.write("\n## Chi tiet cac case FAIL\n\n")
    fails = [r for r in results if not r["passed"]]
    if not fails:
        rf.write("Khong co FAIL.\n\n")
        return
    for fc in fails:
        rf.write(f"### FAIL `{fc['id']}` - {fc['name']} (`{fc['use_case']}`)\n\n")
        rf.write(f"* **Overall**: {fc['overall']}/100\n")
        rf.write(f"* **Per-TC**:\n")
        _write_per_tc_block(rf, fc.get('per_tc'))
        rf.write(f"* **Vi pham**: {fc.get('violations', [])}\n")
        rf.write(f"* **Tom tat**: {fc['reasoning']}\n")
        rf.write(f"* **Eval notes**: {fc['eval_notes']}\n")
        rf.write(f"* **Output**:\n```text\n{fc['output']}\n```\n\n")


def _write_all_responses(rf, results):
    rf.write("## Chi tiet toan bo responses\n\n")
    for r in results:
        sym = "PASS" if r["passed"] else "FAIL"
        rf.write(f"### `{r['id']}` - {r['name']} ({sym} -- {r['overall']}/100)\n\n")
        rf.write(
            f"* **UC**: `{r['use_case']}` | **Criteria**: {r['criteria']} | "
            f"**Latency**: {r['latency']:.2f}s | **Tokens**: In {r['prompt_tokens']} / Out {r['completion_tokens']}\n"
        )
        rf.write("* **Per-TC**:\n")
        per_tc = r.get('per_tc') or {}
        if per_tc:
            _write_per_tc_block(rf, per_tc, score_suffix="/100")
        else:
            rf.write("  * (judge không trả per_tc breakdown)\n")
        rf.write("* **Hoi thoai**:\n")
        for t in r.get("turns", []):
            role = t.get("role", "user").capitalize()
            content = t.get("content", "").replace("\n", "\n  ")
            rf.write(f"  * **{role}**: {content}\n")
        rf.write(f"* **Model tra loi**:\n```text\n{r['output']}\n```\n")
        rf.write(f"* **Tom tat judge**: {r['reasoning']}\n")
        if r.get("judge_thinking"):
            rf.write(f"* **Suy nghi cua Judge**:\n```text\n{r['judge_thinking']}\n```\n")
        if r.get("violations"):
            rf.write(f"* **Vi pham**: {r['violations']}\n")
        rf.write(f"* **Eval notes**: {r['eval_notes']}\n\n---\n\n")


def write_report(test_cases, results, passed_count, total_latency, total_prompt,
                  total_completion, total_score, target_meta=None,
                  judge_model=None, report_path=None):
    n = len(test_cases) or 1
    avg_latency = total_latency / n
    avg_overall = total_score / n
    target_meta = target_meta or {}
    production_count = sum(1 for r in results if r.get("overall", 0) >= PRODUCTION_THRESHOLD)
    per_tc = _compute_per_tc_scores(results)

    _print_console_summary(n, passed_count, production_count, avg_overall, avg_latency,
                            total_prompt, total_completion, target_meta, per_tc,
                            judge_model=judge_model)

    report_path = report_path or REPORT_PATH
    with open(report_path, "w", encoding="utf-8") as rf:
        _write_header(rf, n, passed_count, production_count, avg_overall, avg_latency,
                       total_prompt, total_completion, target_meta, judge_model=judge_model)
        _write_per_tc_table(rf, per_tc)
        _write_summary_table(rf, results)
        _write_fail_details(rf, results)
        _write_all_responses(rf, results)

    print(f"\nBao cao: {report_path}")
