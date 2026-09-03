"""
test_evaluation.py — Programmatic RAG Pipeline Test & Evaluation
=================================================================
Runs 20+ test questions through the full RAG pipeline, logging:
  - Question category
  - The user query
  - Retrieved source documents
  - The LLM-generated answer
  - Whether the answer is the "not available" sentinel
  - Pass/Fail based on heuristic checks (no hard-coded expected answers)

Usage:
  python test_evaluation.py

Output:
  - Console log with results
  - test_results.json (detailed machine-readable log)
  - test_report.txt (human-readable summary)
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime

import rag_pipeline as rag

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)  # Suppress pipeline debug logs during tests

# ---------------------------------------------------------------------------
# Test Question Bank
# ---------------------------------------------------------------------------
# Each question is a dict with:
#   question    : str  — the query
#   category    : str  — one of: direct_factual | cross_document | paraphrased | out_of_kb
#   description : str  — brief explanation of what this test checks
#   expect_no_info: bool — True if we expect the "not available" response

TEST_QUESTIONS = [
    # ── Direct Factual (from a single document) ──────────────────────────────
    {
        "id": "T01",
        "category": "direct_factual",
        "question": "When was Nexora Technologies founded, where is its headquarters, and approximately how many employees does it have?",
        "description": "Tests retrieval of company founding, HQ, and employee count from company_profile.md",
        "expect_no_info": False,
    },
    {
        "id": "T02",
        "category": "direct_factual",
        "question": "What are the three core software products offered by Nexora Technologies?",
        "description": "Tests retrieval of product list from products.md",
        "expect_no_info": False,
    },
    {
        "id": "T03",
        "category": "direct_factual",
        "question": "How much does the Nexora Flow Starter plan cost and what does it include?",
        "description": "Tests retrieval of specific plan price and features from pricing.md",
        "expect_no_info": False,
    },
    {
        "id": "T04",
        "category": "direct_factual",
        "question": "How many days of paid annual leave and casual/sick leave do Nexora employees receive?",
        "description": "Tests retrieval of leave entitlements from hr_policies.md",
        "expect_no_info": False,
    },
    {
        "id": "T05",
        "category": "direct_factual",
        "question": "What is the starting price for Nexora Insight and who are its typical users?",
        "description": "Tests retrieval of Nexora Insight details from products.md",
        "expect_no_info": False,
    },
    {
        "id": "T06",
        "category": "direct_factual",
        "question": "What are the standard working hours at Nexora Technologies?",
        "description": "Tests retrieval of working hours from hr_policies.md",
        "expect_no_info": False,
    },
    {
        "id": "T07",
        "category": "direct_factual",
        "question": "What is the annual learning allowance for Nexora employees?",
        "description": "Tests retrieval of learning and development budget from hr_policies.md",
        "expect_no_info": False,
    },
    {
        "id": "T08",
        "category": "direct_factual",
        "question": "What are the core values of Nexora Technologies?",
        "description": "Tests retrieval of core values table from company_profile.md",
        "expect_no_info": False,
    },
    # ── Cross-Document (requires info from 2+ docs) ───────────────────────────
    {
        "id": "T09",
        "category": "cross_document",
        "question": "What is Nexora Assist, and what are the pricing plans for it?",
        "description": "Product description from products.md and plan pricing from pricing.md",
        "expect_no_info": False,
    },
    {
        "id": "T10",
        "category": "cross_document",
        "question": "What is the typical engagement duration for Cloud Modernization and AI Solutions services?",
        "description": "Engagement durations from services.md",
        "expect_no_info": False,
    },
    {
        "id": "T11",
        "category": "cross_document",
        "question": "What is Nexora Flow designed for, and how much does the Business plan cost?",
        "description": "Description from products.md, Business plan price from pricing.md",
        "expect_no_info": False,
    },
    {
        "id": "T12",
        "category": "cross_document",
        "question": "What are the commercial terms and required notice period for cancelling a month-to-month subscription?",
        "description": "Commercial terms from pricing.md",
        "expect_no_info": False,
    },
    {
        "id": "T13",
        "category": "cross_document",
        "question": "What professional services does Nexora provide, and how does service delivery normally begin?",
        "description": "Services and delivery methodology from services.md",
        "expect_no_info": False,
    },
    # ── Paraphrased (rewording of KB content) ────────────────────────────────
    {
        "id": "T14",
        "category": "paraphrased",
        "question": "Can staff at Nexora work from home, and how many days a week are allowed?",
        "description": "Paraphrase of the hybrid/remote work policy in hr_policies.md",
        "expect_no_info": False,
    },
    {
        "id": "T15",
        "category": "paraphrased",
        "question": "How many weeks does a standard SaaS implementation rollout usually require for clients?",
        "description": "Paraphrase of implementation timeline from faqs.md",
        "expect_no_info": False,
    },
    {
        "id": "T16",
        "category": "paraphrased",
        "question": "Can prospective clients test the software before purchasing? Is there a trial period?",
        "description": "Paraphrase of 14-day free trial question from faqs.md",
        "expect_no_info": False,
    },
    {
        "id": "T17",
        "category": "paraphrased",
        "question": "If a customer needs custom software capabilities or integrations, does Nexora support that?",
        "description": "Paraphrase of custom features and software development from faqs.md and services.md",
        "expect_no_info": False,
    },
    {
        "id": "T18",
        "category": "paraphrased",
        "question": "How often are performance reviews conducted for employees at Nexora?",
        "description": "Paraphrase of performance reviews from hr_policies.md",
        "expect_no_info": False,
    },
    # ── Out-of-KB (answers not in the knowledge base) ───────────────────────
    {
        "id": "T19",
        "category": "out_of_kb",
        "question": "What is the current stock price of Nexora Technologies on the Bombay Stock Exchange?",
        "description": "Nexora is private; stock price not in KB",
        "expect_no_info": True,
    },
    {
        "id": "T20",
        "category": "out_of_kb",
        "question": "What was the total annual revenue of Nexora Technologies in FY 2024?",
        "description": "Revenue figure is not in KB",
        "expect_no_info": True,
    },
    {
        "id": "T21",
        "category": "out_of_kb",
        "question": "What is the capital city of Australia?",
        "description": "Completely unrelated to company KB",
        "expect_no_info": True,
    },
    {
        "id": "T22",
        "category": "out_of_kb",
        "question": "Can you write me a quicksort algorithm in Python?",
        "description": "Coding query completely unrelated to KB",
        "expect_no_info": True,
    },
]


# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------

def check_pass(result: dict, expect_no_info: bool) -> tuple[bool, str]:
    """
    Heuristic pass/fail check:
    - If expect_no_info=True: PASS if answer contains the sentinel phrase
    - If expect_no_info=False: PASS if answer does NOT contain sentinel and is non-empty
    """
    answer = result.get("answer", "")
    has_sentinel = rag.NO_INFO_RESPONSE.lower() in answer.lower()

    if expect_no_info:
        passed = has_sentinel
        reason = "Correctly returned 'not available'" if passed else "Expected 'not available' but got a real answer"
    else:
        passed = not has_sentinel and len(answer.strip()) > 20
        reason = "Answered from KB" if passed else (
            "Returned 'not available' — may indicate retrieval miss" if has_sentinel
            else "Answer too short"
        )

    return passed, reason


def run_evaluation():
    """Run all test questions and produce a detailed report."""

    print("=" * 70)
    print("  NovaTech KB Chatbot — RAG Pipeline Evaluation")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Initialize pipeline
    print("\n▶ Initializing RAG pipeline...")
    try:
        vectorstore, llm = rag.initialize_pipeline(force_rebuild=False)
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return

    print(f"✓ Pipeline ready. Running {len(TEST_QUESTIONS)} test questions...\n")

    results = []
    passed_count = 0
    failed_count = 0

    category_stats: dict[str, dict] = {}

    for i, test in enumerate(TEST_QUESTIONS, start=1):
        tid = test["id"]
        category = test["category"]
        question = test["question"]
        expect_no_info = test["expect_no_info"]

        print(f"[{i:02d}/{len(TEST_QUESTIONS)}] {tid} | {category.upper()}")
        print(f"     Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        start_time = time.time()
        try:
            result = rag.answer_question(question, vectorstore, llm)
            elapsed = time.time() - start_time

            passed, reason = check_pass(result, expect_no_info)

            answer_preview = result["answer"][:120].replace("\n", " ")
            sources_list = [s["file"] for s in result["sources"]]

            status_icon = "✓ PASS" if passed else "✗ FAIL"
            print(f"     {status_icon} | {reason} | {elapsed:.2f}s")
            print(f"     A: {answer_preview}...")
            print(f"     Sources: {', '.join(sources_list) if sources_list else 'None'}")
            print()

            if passed:
                passed_count += 1
            else:
                failed_count += 1

            # Category stats
            if category not in category_stats:
                category_stats[category] = {"total": 0, "passed": 0}
            category_stats[category]["total"] += 1
            if passed:
                category_stats[category]["passed"] += 1

            results.append({
                "id": tid,
                "category": category,
                "description": test["description"],
                "question": question,
                "expect_no_info": expect_no_info,
                "answer": result["answer"],
                "sources": result["sources"],
                "passed": passed,
                "reason": reason,
                "elapsed_seconds": round(elapsed, 3),
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"     ✗ ERROR | {e} | {elapsed:.2f}s\n")
            failed_count += 1
            results.append({
                "id": tid,
                "category": category,
                "description": test["description"],
                "question": question,
                "expect_no_info": expect_no_info,
                "answer": f"ERROR: {e}",
                "sources": [],
                "passed": False,
                "reason": f"Exception: {e}",
                "elapsed_seconds": round(elapsed, 3),
            })

    # ── Summary ──
    total = len(TEST_QUESTIONS)
    pass_rate = (passed_count / total) * 100

    print("=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total tests : {total}")
    print(f"  Passed      : {passed_count}  ✓")
    print(f"  Failed      : {failed_count}  ✗")
    print(f"  Pass rate   : {pass_rate:.1f}%")
    print()
    print("  Results by Category:")
    for cat, stats in sorted(category_stats.items()):
        cat_pass_rate = (stats["passed"] / stats["total"]) * 100
        print(f"    {cat:<20} {stats['passed']}/{stats['total']} ({cat_pass_rate:.0f}%)")
    print("=" * 70)

    # ── Write JSON Results ──
    output_json = Path("test_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_timestamp": datetime.now().isoformat(),
                "model": rag.LLM_MODEL,
                "embedding_model": rag.EMBEDDING_MODEL_NAME,
                "total": total,
                "passed": passed_count,
                "failed": failed_count,
                "pass_rate_pct": round(pass_rate, 1),
                "category_stats": category_stats,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  Detailed results saved to: {output_json}")

    # ── Write Text Report ──
    output_txt = Path("test_report.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("NovaTech KB Chatbot — RAG Evaluation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {rag.LLM_MODEL}\n")
        f.write(f"Embedding: {rag.EMBEDDING_MODEL_NAME}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"PASS RATE: {pass_rate:.1f}% ({passed_count}/{total})\n\n")

        f.write("RESULTS BY CATEGORY:\n")
        for cat, stats in sorted(category_stats.items()):
            cat_rate = (stats["passed"] / stats["total"]) * 100
            f.write(f"  {cat}: {stats['passed']}/{stats['total']} ({cat_rate:.0f}%)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED RESULTS:\n\n")

        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            f.write(f"[{r['id']}] {status} | {r['category'].upper()}\n")
            f.write(f"Description : {r['description']}\n")
            f.write(f"Question    : {r['question']}\n")
            f.write(f"Expect N/A  : {r['expect_no_info']}\n")
            f.write(f"Reason      : {r['reason']}\n")
            f.write(f"Elapsed     : {r['elapsed_seconds']}s\n")
            f.write(f"Answer      : {r['answer'][:300]}...\n")
            sources_str = ", ".join([s["file"] for s in r.get("sources", [])]) or "None"
            f.write(f"Sources     : {sources_str}\n")
            f.write("-" * 70 + "\n")

    print(f"  Human-readable report saved to: {output_txt}\n")
    return results


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
