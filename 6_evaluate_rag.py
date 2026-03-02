"""
STEP 6: 엔드투엔드 RAG 시스템 평가
논문 Section III-D "Answer Correctness of the RAG System" — Table III 재현
원본: evaluation/evaluation_QA_system/evaluation_pipeline.ipynb
      evaluation/evaluation_QA_system/RAG_evaluator.py

평가 방법 (논문 원본 그대로):
  - BioASQ yesno 질문으로 RAG 시스템의 정답률 측정
  - RAG_evaluator.py의 handle_yesno() → yesno_eval() 로직 사용
  - sklearn의 accuracy, recall, precision, f1 (weighted) 계산

비교 대상:
  - GPT-3.5-turbo + BM25    (retriever=2, question_type=2)
  - GPT-3.5-turbo + Hybrid  (retriever=3, question_type=2)
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

from bm25_retriever import BM25Retriever
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

# 평가 질문 수 제한 (API 비용 절약)
# 논문: yesno 87개 전체 사용. 실습: 조절 가능
MAX_QUESTIONS = None  # None이면 전체, 숫자면 해당 수만큼
if MAX_QUESTIONS:
    yesno_questions = yesno_questions[:MAX_QUESTIONS]
    print(f"  → 실습용으로 {MAX_QUESTIONS}개만 평가")


# ─── 2. 원본 RAG_evaluator.py의 핵심 평가 함수들 ───

def yesno_eval(response, true_answer):
    """
    원본 RAG_evaluator.py의 yesno_eval() 그대로
    응답에서 yes/no를 추출하여 정답과 비교
    """
    response_lower = response.lower().strip()
    true_lower = true_answer.lower().strip()

    if "yes" in response_lower:
        pred = "yes"
    elif "no" in response_lower:
        pred = "no"
    else:
        pred = response_lower

    return pred == true_lower, pred


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


# ─── 3. RAG 평가 함수 (원본 evaluation_pipeline.ipynb 워크플로) ───

def evaluate_rag(retriever, retriever_name, chat, questions, k=10):
    """
    원본 evaluation_pipeline.ipynb의 워크플로:
    1. MedRAG.get_answer(question) → 검색 + 생성
    2. RAG_evaluator.handle_yesno() → yesno_eval()
    3. RAG_evaluator.analyze_performance() → 메트릭 계산
    """
    print(f"\n{'='*60}")
    print(f"RAG 평가: {retriever_name}")
    print(f"질문 수: {len(questions)}개")
    print(f"{'='*60}")

    results = []
    y_true = []
    y_pred = []
    recall_retrieval_list = []
    precision_rag_list = []

    for q in tqdm(questions, desc=f"{retriever_name}"):
        # ── 검색 (원본 med_rag.py의 get_answer 중 검색 부분) ──
        start_retrieval = time.time()
        try:
            if hasattr(retriever, 'reranker'):
                retrieved_docs = json.loads(
                    retriever.retrieve_docs(q['body'], top_n=10, k=k)
                )
            else:
                retrieved_docs = json.loads(
                    retriever.retrieve_docs(q['body'], k=k)
                )
        except Exception as e:
            continue
        retrieval_time = time.time() - start_retrieval

        if not retrieved_docs:
            y_true.append(q['exact_answer'].lower())
            y_pred.append("no_docs_found")
            continue

        # 검색된 PMID (원본 med_rag.py의 extract_pmids)
        k_pubmedids = [str(doc['PMID']) for doc in retrieved_docs.values()]

        # ── 생성 (원본 openAI_chat.py의 create_chat) ──
        start_generation = time.time()
        try:
            answer_json = chat.create_chat(q['body'], retrieved_docs)
            answer = json.loads(answer_json)
            response = answer.get('response', '')
            used_pmids = list(map(str, answer.get('used_PMIDs', [])))
        except Exception as e:
            continue
        generation_time = time.time() - start_generation

        # ── 평가: yesno (원본 RAG_evaluator.py의 handle_yesno) ──
        is_correct, pred = yesno_eval(response, q['exact_answer'])

        y_true.append(q['exact_answer'].lower())
        y_pred.append(pred)

        # ── PMID 매칭 (원본 RAG_evaluator.py의 compare_pubmed_ids) ──
        ground_truth_urls = q.get('documents', [])
        ground_truth_pmids = extract_pubmedid(ground_truth_urls)

        # Retriever recall (원본 analyze_performance)
        _, num_matched_retrieved, _ = compare_pubmed_ids(k_pubmedids, ground_truth_urls)
        recall_retrieval = (
            num_matched_retrieved / len(ground_truth_pmids)
            if ground_truth_pmids else 0
        )
        recall_retrieval_list.append(recall_retrieval)

        # RAG used precision (원본 analyze_performance)
        _, num_matched_used, _ = compare_pubmed_ids(used_pmids, ground_truth_urls)
        precision_rag = (
            num_matched_used / len(used_pmids)
            if used_pmids else 0
        )
        precision_rag_list.append(precision_rag)

        results.append({
            "question": q['body'],
            "true_answer": q['exact_answer'].lower(),
            "rag_response": response.lower(),
            "predicted": pred,
            "correct": is_correct,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
        })

    # ── 메트릭 계산 (원본 RAG_evaluator.py의 analyze_performance) ──
    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    else:
        accuracy = recall = precision = f1 = 0

    avg_recall_retrieval = np.mean(recall_retrieval_list) if recall_retrieval_list else 0
    avg_precision_rag = np.mean(precision_rag_list) if precision_rag_list else 0

    # No Docs Found 통계 (원본 analyze_performance)
    no_docs = sum(1 for p in y_pred if p == "no_docs_found")
    no_docs_pct = (no_docs / len(y_pred) * 100) if y_pred else 0

    print(f"\n  Metrics - RAG Q&A (원본 analyze_performance 형식):")
    print(f"  Total Questions: {len(y_true)}")
    print(f"  Accuracy:  {accuracy:.2f}")
    print(f"  Recall:    {recall:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  F1 Score:  {f1:.2f}")
    print(f"  No Docs Found: {no_docs} ({no_docs_pct:.1f}%)")
    print(f"  Retriever Avg Recall: {avg_recall_retrieval:.3f}")
    print(f"  RAG Used Avg Precision: {avg_precision_rag:.3f}")

    return {
        "retriever": retriever_name,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "retriever_recall": avg_recall_retrieval,
        "rag_precision": avg_precision_rag,
        "n_questions": len(y_true),
    }


# ─── 4. 실행 ───
# 원본 evaluation_pipeline.ipynb: question_type=2 (yesno)
print("Chat 모델 초기화 (GPT-3.5-turbo, yesno)...")
chat = Chat(question_type=2)

print("\nBM25 리트리버 초기화...")
bm25 = BM25Retriever()
bm25_result = evaluate_rag(bm25, "BM25", chat, yesno_questions, k=10)

print("\nHybrid 리트리버 초기화...")
hybrid = HybridRetriever()
hybrid_result = evaluate_rag(hybrid, "Hybrid (BM25+MedCPT)", chat, yesno_questions, k=50)


# ─── 5. 논문 Table III 형식으로 출력 ───
print("\n" + "=" * 70)
print("논문 Table III 재현 — 엔드투엔드 RAG 시스템 성능 (yesno)")
print("=" * 70)
print(f"{'RAG with Retriever':<30} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'F1':<10}")
print("-" * 70)
for r in [bm25_result, hybrid_result]:
    print(f"{r['retriever']:<30} {r['accuracy']:<10.2f} {r['recall']:<10.2f} "
          f"{r['precision']:<10.2f} {r['f1']:<10.2f}")
print("-" * 70)
print("논문 결과 (참고):")
print(f"{'GPT-3.5 / BM25':<30} {'0.72':<10} {'0.72':<10} {'0.83':<10} {'0.74':<10}")
print(f"{'GPT-3.5 / Hybrid':<30} {'0.86':<10} {'0.86':<10} {'0.89':<10} {'0.86':<10}")
print(f"\n평가 질문 수: {bm25_result['n_questions']}개 (BioASQ yesno)")
print(f"※ 논문 결론: Hybrid retriever가 모든 메트릭에서 최고 성능")