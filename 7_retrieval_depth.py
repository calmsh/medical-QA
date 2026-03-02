"""
STEP 7: Retrieval Depth 분석
논문 Section IV-A "Effect of Retrieval Depth on Performance" — Table IV 재현
원본: evaluation/evaluation_QA_system/evaluation_pipeline.ipynb
      (Hybrid retriever의 k 파라미터를 변경하며 실험)

핵심 분석:
  - BM25로 검색할 문서 수(k)를 20, 50, 100으로 변경
  - MedCPT Cross-Encoder로 리랭킹 후 top 10 사용
  - k=50이 최적: "Retrieving 50 documents before reranking yields the best accuracy (0.90)"

※ BioASQ yesno 질문 사용 (step7과 동일한 데이터)
"""

import sys
import json
import os
import time
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

sys.path.append("./rag_system")

from hybrid_retriever import HybridRetriever
from openAI_chat import Chat

# ─── 1. BioASQ yesno 질문 로드 ───
YESNO_PATH = "./data/bioasq_filtered/yesno_questions.json"

if not os.path.exists(YESNO_PATH):
    print("오류: BioASQ yesno 질문 데이터가 없습니다!")
    print("  → step4_preprocess_bioasq.py를 먼저 실행하세요.")
    exit(1)

with open(YESNO_PATH, 'r', encoding='utf-8') as f:
    yesno_data = json.load(f)

yesno_questions = yesno_data['questions']
print(f"BioASQ yesno 질문: {len(yesno_questions)}개")

# 평가 질문 수 제한 (API 비용 절약: depth 3개 × 질문 수 × API 호출)
MAX_QUESTIONS = None  # None이면 전체, 숫자면 해당 수만큼
if MAX_QUESTIONS:
    yesno_questions = yesno_questions[:MAX_QUESTIONS]
    print(f"  → 실습용으로 {MAX_QUESTIONS}개만 평가")


# ─── 2. 평가 함수 (원본 RAG_evaluator.py 기반) ───

def yesno_eval(response, true_answer):
    """원본 RAG_evaluator.py의 yesno_eval()"""
    response_lower = response.lower().strip()
    if "yes" in response_lower:
        pred = "yes"
    elif "no" in response_lower:
        pred = "no"
    else:
        pred = response_lower
    return pred == true_answer.lower().strip(), pred


def extract_pubmedid(documents):
    """원본 RAG_evaluator.py의 extract_pubmedid()"""
    return [
        re.search(r"pubmed/(\d+)", doc).group(1)
        for doc in documents
        if re.search(r"pubmed/(\d+)", doc)
    ]


def compare_pubmed_ids(retrieved_pmids, ground_truth_urls):
    """원본 RAG_evaluator.py의 compare_pubmed_ids()"""
    if not isinstance(retrieved_pmids, list):
        retrieved_pmids = []
    extracted_ids = [
        re.search(r"pubmed/(\d+)", doc).group(1)
        for doc in ground_truth_urls
        if re.search(r"pubmed/(\d+)", doc)
    ]
    matched_ids = [pid for pid in extracted_ids if pid in retrieved_pmids]
    return bool(matched_ids), len(matched_ids), matched_ids


# ─── 3. Retrieval Depth 설정 (논문 Table IV와 동일) ───
retrieval_depths = [20, 50, 100]  # 논문 Table IV의 k 값
top_n = 10                         # 논문: "reranking applied to the top 10 documents"

print("\nHybrid 리트리버 초기화...")
hybrid = HybridRetriever()

print("Chat 모델 초기화 (GPT-3.5-turbo, yesno)...")
chat = Chat(question_type=2)

results_table = []

# ─── 4. 각 depth별 평가 실행 ───
for depth in retrieval_depths:
    print(f"\n{'='*60}")
    print(f"Retrieval Depth: k={depth} (top_n={top_n} 리랭킹)")
    print(f"{'='*60}")

    y_true = []
    y_pred = []
    retrieval_times = []
    generation_times = []
    total_times = []
    recall_retrieval_list = []

    for q in tqdm(yesno_questions, desc=f"k={depth}"):
        start_total = time.time()

        # ── 검색 + 리랭킹 (원본 hybrid_retriever.py) ──
        start_retrieval = time.time()
        try:
            results = json.loads(
                hybrid.retrieve_docs(q['body'], top_n=top_n, k=depth)
            )
        except Exception as e:
            continue
        retrieval_time = time.time() - start_retrieval
        retrieval_times.append(retrieval_time)

        if not results:
            y_true.append(q['exact_answer'].lower())
            y_pred.append("no_docs_found")
            total_times.append(time.time() - start_total)
            continue

        # PMID 매칭 (Retriever recall)
        k_pubmedids = [str(doc['PMID']) for doc in results.values()]
        ground_truth_urls = q.get('documents', [])
        ground_truth_pmids = extract_pubmedid(ground_truth_urls)
        _, num_matched, _ = compare_pubmed_ids(k_pubmedids, ground_truth_urls)
        recall_retrieval = (
            num_matched / len(ground_truth_pmids) if ground_truth_pmids else 0
        )
        recall_retrieval_list.append(recall_retrieval)

        # ── 생성 (원본 openAI_chat.py) ──
        start_generation = time.time()
        try:
            answer_json = chat.create_chat(q['body'], results)
            answer = json.loads(answer_json)
            response = answer.get('response', '')
        except Exception as e:
            continue
        generation_time = time.time() - start_generation
        generation_times.append(generation_time)

        total_time = time.time() - start_total
        total_times.append(total_time)

        # ── yesno 평가 ──
        _, pred = yesno_eval(response, q['exact_answer'])
        y_true.append(q['exact_answer'].lower())
        y_pred.append(pred)

    # ── 메트릭 계산 ──
    if y_true:
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    else:
        accuracy = recall = precision = f1 = 0

    avg_retrieval = np.mean(retrieval_times) if retrieval_times else 0
    std_retrieval = np.std(retrieval_times) if retrieval_times else 0
    avg_generation = np.mean(generation_times) if generation_times else 0
    std_generation = np.std(generation_times) if generation_times else 0
    avg_total = np.mean(total_times) if total_times else 0
    std_total = np.std(total_times) if total_times else 0
    avg_recall_retrieval = np.mean(recall_retrieval_list) if recall_retrieval_list else 0

    results_table.append({
        "docs": depth,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "retriever_recall": avg_recall_retrieval,
        "retrieval_time": f"{avg_retrieval:.2f} ± {std_retrieval:.2f}",
        "generation_time": f"{avg_generation:.2f} ± {std_generation:.2f}",
        "total_time": f"{avg_total:.2f} ± {std_total:.2f}",
        "n_questions": len(y_true),
    })

    print(f"\n  k={depth} 결과:")
    print(f"  Accuracy={accuracy:.2f}, Recall={recall:.2f}, "
          f"Precision={precision:.2f}, F1={f1:.2f}")
    print(f"  Retriever Recall: {avg_recall_retrieval:.3f}")
    print(f"  Retrieval: {avg_retrieval:.2f}s ± {std_retrieval:.2f}s")
    print(f"  Generation: {avg_generation:.2f}s ± {std_generation:.2f}s")
    print(f"  Total: {avg_total:.2f}s ± {std_total:.2f}s")


# ─── 5. 논문 Table IV 형식으로 출력 ───
print("\n" + "=" * 90)
print("논문 Table IV 재현 — Retrieval Depth별 성능 비교")
print("=" * 90)
print(f"{'Docs':<6} {'Acc':<8} {'Recall':<8} {'Prec':<8} {'F1':<8} "
      f"{'Retrieval Time(s)':<20} {'Total Time(s)':<20}")
print("-" * 78)
for r in results_table:
    print(f"{r['docs']:<6} {r['accuracy']:<8.2f} {r['recall']:<8.2f} "
          f"{r['precision']:<8.2f} {r['f1']:<8.2f} "
          f"{r['retrieval_time']:<20} {r['total_time']:<20}")
print("-" * 78)
print("논문 결과 (참고):")
print(f"{'20':<6} {'0.89':<8} {'0.89':<8} {'0.90':<8} {'0.88':<8} "
      f"{'0.39 ± 0.07':<20} {'1.52 ± 0.42':<20}")
print(f"{'50':<6} {'0.90':<8} {'0.90':<8} {'0.91':<8} {'0.90':<8} "
      f"{'0.82 ± 0.13':<20} {'1.91 ± 0.36':<20}")
print(f"{'100':<6} {'0.87':<8} {'0.87':<8} {'0.88':<8} {'0.87':<8} "
      f"{'1.54 ± 0.16':<20} {'2.62 ± 0.44':<20}")

print(f"\n평가 질문 수: {results_table[0]['n_questions']}개 (BioASQ yesno)")
print(f"※ 논문 핵심 발견: k=50이 최적 (accuracy=0.90)")
print(f"  k=100은 오히려 성능 하락 → 관련성 낮은 문서가 노이즈로 작용")
print(f"※ BM25 검색 시간은 일정(82ms), MedCPT 리랭킹이 k에 비례하여 증가")