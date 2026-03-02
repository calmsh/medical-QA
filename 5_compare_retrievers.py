"""
STEP 5: 성능 비교
논문 Section III-C "Document Relevancy" — Table II 재현
원본: evaluation/evaluation_QA_system/RAG_evaluator.py의 compare_pubmed_ids()
      evaluation/evaluation_QA_system/evaluation_pipeline.ipynb

평가 방법 (논문 원본 그대로):
  - BioASQ 질문의 ground truth PMID와 리트리버가 검색한 PMID 비교
  - Recall = 매칭된 PMID 수 / ground truth PMID 수
  - Precision = 매칭된 PMID 수 / RAG가 사용한 PMID 수

비교 대상:
  1. BM25 (Elasticsearch)         — 원본 retriever=2
  2. Hybrid (BM25 + MedCPT 리랭킹) — 원본 retriever=3

※ BioBERT(retriever=1), MedCPT(retriever=4)는 사전 임베딩 + FAISS 서버 필요
  → 논문의 핵심 비교는 BM25 vs Hybrid이므로 이 둘로 충분
"""

import sys
import json
import os
import time
import re
import numpy as np
from tqdm import tqdm

sys.path.append("./rag_system")

from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever

# ─── 1. BioASQ 필터링 데이터 로드 ───
BIOASQ_PATH = "./data/bioasq_filtered/all_filtered_questions.json"

if not os.path.exists(BIOASQ_PATH):
    print("오류: BioASQ 필터링 데이터가 없습니다!")
    print("  → step4_preprocess_bioasq.py를 먼저 실행하세요.")
    exit(1)

with open(BIOASQ_PATH, 'r', encoding='utf-8') as f:
    bioasq_data = json.load(f)

all_questions = bioasq_data['questions']
print(f"BioASQ 필터링된 질문: {len(all_questions)}개")

# 평가할 질문 수 제한 (전체 평가는 시간이 오래 걸림)
# 논문: 723개 전체 사용. 실습: 시간 절약을 위해 조절 가능
MAX_QUESTIONS = None  # None이면 전체 사용, 숫자면 해당 수만큼만
if MAX_QUESTIONS:
    all_questions = all_questions[:MAX_QUESTIONS]
    print(f"  → 실습용으로 {MAX_QUESTIONS}개만 평가")


# ─── 2. 원본 RAG_evaluator.py의 핵심 함수들 ───

def extract_pubmedid(documents):
    """원본 RAG_evaluator.py의 extract_pubmedid() 그대로"""
    return [
        re.search(r"pubmed/(\d+)", doc).group(1)
        for doc in documents
        if re.search(r"pubmed/(\d+)", doc)
    ]


def compare_pubmed_ids(retrieved_pmids, ground_truth_urls):
    """원본 RAG_evaluator.py의 compare_pubmed_ids() 그대로"""
    if not isinstance(retrieved_pmids, list):
        retrieved_pmids = []

    extracted_ids = [
        re.search(r"pubmed/(\d+)", doc).group(1)
        for doc in ground_truth_urls
        if re.search(r"pubmed/(\d+)", doc)
    ]

    matched_ids = [pid for pid in extracted_ids if pid in retrieved_pmids]

    return bool(matched_ids), len(matched_ids), matched_ids


# ─── 3. 리트리버 평가 함수 ───
# 원본 evaluation_pipeline.ipynb + RAG_evaluator.py의 handle_yesno/handle_summary_factoid 로직

def evaluate_retriever(retriever, retriever_name, questions, k=10):
    """
    논문 Table II 재현: 리트리버의 Recall/Precision 측정

    원본 RAG_evaluator.py의 analyze_performance()에서:
    - recall = matching_retrieved_ids / ground_truth_pmids  (per question)
    - precision = matching_used_ids / used_pmids  (per question)

    여기서는 retriever만 평가 (LLM 생성 없이):
    - recall = 매칭된 PMID / ground truth PMID
    - precision = 매칭된 PMID / 검색된 PMID
    """
    print(f"\n{'='*60}")
    print(f"리트리버: {retriever_name} (k={k})")
    print(f"질문 수: {len(questions)}개")
    print(f"{'='*60}")

    recall_list = []
    precision_list = []
    retrieval_times = []
    no_docs_count = 0

    for q in tqdm(questions, desc=f"{retriever_name} 평가"):
        ground_truth_urls = q.get('documents', [])
        ground_truth_pmids = extract_pubmedid(ground_truth_urls)

        if not ground_truth_pmids:
            continue

        # 검색 수행
        start = time.time()
        try:
            if hasattr(retriever, 'reranker'):
                # Hybrid: retrieve_docs(query, top_n, k)
                results = json.loads(retriever.retrieve_docs(q['body'], top_n=10, k=k))
            else:
                # BM25: retrieve_docs(query, k)
                results = json.loads(retriever.retrieve_docs(q['body'], k=k))
        except Exception as e:
            continue
        elapsed = time.time() - start
        retrieval_times.append(elapsed)

        if not results:
            no_docs_count += 1
            recall_list.append(0)
            precision_list.append(0)
            continue

        # 검색된 PMID 추출 (원본 med_rag.py의 extract_pmids)
        retrieved_pmids = [str(doc['PMID']) for doc in results.values()]

        # 원본 RAG_evaluator.py의 compare_pubmed_ids() 로직
        _, num_matched, matched_ids = compare_pubmed_ids(
            retrieved_pmids, ground_truth_urls
        )

        # Recall: 매칭된 / ground truth (원본 analyze_performance의 recall_retrieval)
        recall = num_matched / len(ground_truth_pmids) if ground_truth_pmids else 0
        # Precision: 매칭된 / 검색된 (원본 analyze_performance의 precision_rag)
        precision = num_matched / len(retrieved_pmids) if retrieved_pmids else 0

        recall_list.append(recall)
        precision_list.append(precision)

    avg_recall = np.mean(recall_list) if recall_list else 0
    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_time = np.mean(retrieval_times) if retrieval_times else 0
    std_time = np.std(retrieval_times) if retrieval_times else 0

    print(f"\n결과 ({retriever_name}):")
    print(f"  평가 질문 수:  {len(recall_list)}")
    print(f"  평균 Recall:   {avg_recall:.3f}")
    print(f"  평균 Precision: {avg_precision:.3f}")
    print(f"  평균 응답시간:  {avg_time:.2f}s ± {std_time:.2f}s")
    print(f"  No Docs Found: {no_docs_count}건")

    return {
        "retriever": retriever_name,
        "recall": avg_recall,
        "precision": avg_precision,
        "avg_time": avg_time,
        "std_time": std_time,
        "n_questions": len(recall_list),
    }


# ─── 4. 리트리버 초기화 및 평가 실행 ───
print("\nBM25 리트리버 초기화...")
bm25 = BM25Retriever()

print("Hybrid 리트리버 초기화 (MedCPT Cross-Encoder 로딩 중)...")
hybrid = HybridRetriever()

# 평가 실행
# BM25: k=10 (논문 기본값)
bm25_result = evaluate_retriever(bm25, "BM25", all_questions, k=10)

# Hybrid: k=50 (BM25로 50개 검색 → MedCPT로 리랭킹 → top 10)
# 논문 Table IV에서 k=50이 최적
hybrid_result = evaluate_retriever(hybrid, "Hybrid (BM25+MedCPT)", all_questions, k=50)


# ─── 5. 논문 Table II 형식으로 결과 출력 ───
print("\n" + "=" * 80)
print("논문 Table II 재현 — 리트리버 성능 비교")
print("=" * 80)
print(f"{'Retriever':<30} {'Recall':<10} {'Precision':<12} {'Response Time':<20}")
print("-" * 72)

for r in [bm25_result, hybrid_result]:
    print(f"{r['retriever']:<30} {r['recall']:<10.3f} {r['precision']:<12.3f} "
          f"{r['avg_time']:.2f}s ± {r['std_time']:.2f}s")

print("-" * 72)
print("논문 결과 (참고):")
print(f"{'BM25':<30} {'0.537':<10} {'0.322':<12} {'0.08s ± 0.04s':<20}")
print(f"{'BioBERT':<30} {'0.447':<10} {'0.252':<12}")
print(f"{'MedCPT':<30} {'0.423':<10} {'0.243':<12}")
print(f"{'Hybrid (BM25+MedCPT)':<30} {'0.567':<10} {'0.319':<12}")

print(f"\n평가 질문 수: {bm25_result['n_questions']}개 (BioASQ training13b 필터링)")
print(f"※ 논문 결론: Hybrid Retriever가 가장 높은 Recall 달성 (0.567)")
print(f"※ BM25는 빠르지만 의미적 관련성 부족, MedCPT 리랭킹이 이를 보완")