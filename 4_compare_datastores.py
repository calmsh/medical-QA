"""
STEP 4: 데이터 저장소 비교
논문 Section III-B "Indexing and Query Time" — Table I 재현
원본: evaluation/evaluation_data_storages/elasticsearch/eval_elastic.ipynb
      evaluation/evaluation_data_storages/faiss/request.ipynb

비교 대상:
  - Elasticsearch (Sparse, BM25) — 원본: eval_elastic.ipynb의 bm25_search()
  - FAISS (Dense, L2 Distance)  — 원본: request.ipynb의 FlatL2 인덱스

변경점 (원본 대비):
  - BioASQ 필터링된 질문을 테스트 쿼리로 사용 (하드코딩 대신 실제 데이터)
  - Docker → Windows 로컬 Elasticsearch
  - 나머지 로직 동일

※ MongoDB는 Docker 필요 + 논문에서 성능이 가장 낮아 생략
"""

import time
import json
import os
import numpy as np
import faiss
from elasticsearch import Elasticsearch
from tqdm import tqdm

# ─── BioASQ 필터링 데이터에서 테스트 쿼리 로드 ───
BIOASQ_FILTERED_PATH = "./data/bioasq_filtered/all_filtered_questions.json"

if os.path.exists(BIOASQ_FILTERED_PATH):
    with open(BIOASQ_FILTERED_PATH, 'r', encoding='utf-8') as f:
        bioasq_data = json.load(f)
    # 최대 50개 질문을 테스트 쿼리로 사용 (논문과 유사한 규모)
    test_queries = [q['body'] for q in bioasq_data['questions'][:50]]
    print(f"BioASQ 필터링 데이터에서 {len(test_queries)}개 쿼리 로드")
else:
    print("※ BioASQ 필터링 데이터 없음 → 기본 쿼리 사용")
    print("  (step4_preprocess_bioasq.py를 먼저 실행하세요)")
    test_queries = [
        "What is the treatment for diabetes?",
        "Is Alzheimer's disease hereditary?",
        "Do CpG islands colocalise with transcription start sites?",
        "What are the symptoms of COVID-19?",
        "How does immunotherapy work for cancer?",
        "What causes Parkinson's disease?",
        "Is aspirin effective for heart disease prevention?",
        "What are the side effects of chemotherapy?",
        "How is tuberculosis diagnosed?",
        "What is the role of insulin in the body?",
    ]

# ─── 1. Elasticsearch BM25 응답 시간 측정 ───
# 원본: eval_elastic.ipynb의 bm25_search() 함수
print("=" * 60)
print("1. Elasticsearch BM25 응답 시간 측정")
print("   논문 Table I: 82ms ± 37ms")
print("=" * 60)

es = Elasticsearch(['http://localhost:9200'])

es_times = []
for query in tqdm(test_queries, desc="BM25 쿼리"):
    start = time.time()
    # 원본 eval_elastic.ipynb의 bm25_search()와 동일한 쿼리 구조
    es.search(
        index="pubmed_index",
        body={
            "size": 10,
            "query": {"match": {"content": query}},
            "_source": ["PMID", "title", "content"]
        }
    )
    elapsed = (time.time() - start) * 1000  # ms
    es_times.append(elapsed)

print(f"\nElasticsearch BM25 응답 시간:")
print(f"  평균: {np.mean(es_times):.1f}ms ± {np.std(es_times):.1f}ms")
print(f"  논문: 82ms ± 37ms")

# ─── 2. FAISS Dense Vector 검색 시간 측정 ───
# 원본: information_retrieval/faiss_container/server.py → IndexFlatL2 사용
print("\n" + "=" * 60)
print("2. FAISS L2 Distance 응답 시간 측정")
print("   논문 Table I: 657ms ± 127ms")
print("=" * 60)

VECTOR_DIM = 768   # 원본 server.py: BioBERT/MedCPT 벡터 차원
NUM_VECTORS = es.count(index='pubmed_index')['count']  # ES 인덱스와 동일 규모

# 메모리 제한: 2.4M 벡터는 ~7GB 필요. 불가능하면 축소
MAX_VECTORS = 2400000  # 2.4M (RAM 16GB 이상 필요)
if NUM_VECTORS > MAX_VECTORS:
    print(f"  ※ 메모리 제한으로 {MAX_VECTORS:,}개로 축소 (실제: {NUM_VECTORS:,}개)")
    NUM_VECTORS = MAX_VECTORS

print(f"FAISS 인덱스 생성 중 ({NUM_VECTORS:,}개 벡터, {VECTOR_DIM}차원)...")
index_start = time.time()

# 원본 server.py와 동일: FlatL2 인덱스
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
vectors = np.random.rand(NUM_VECTORS, VECTOR_DIM).astype('float32')
faiss_index.add(vectors)

faiss_index_time = time.time() - index_start
print(f"FAISS 인덱싱 시간: {faiss_index_time:.2f}초")

# 검색 시간 측정
faiss_times = []
for _ in tqdm(range(len(test_queries)), desc="FAISS 쿼리"):
    query_vec = np.random.rand(1, VECTOR_DIM).astype('float32')
    start = time.time()
    distances, indices = faiss_index.search(query_vec, 10)
    elapsed = (time.time() - start) * 1000
    faiss_times.append(elapsed)

print(f"\nFAISS L2 응답 시간:")
print(f"  평균: {np.mean(faiss_times):.1f}ms ± {np.std(faiss_times):.1f}ms")
print(f"  논문: 657ms ± 127ms (2.4M 문서 기준)")

# ─── 3. 결과 요약 (논문 Table I 형식) ───
print("\n" + "=" * 60)
print("결과 요약 — 논문 Table I 재현")
print("=" * 60)
print(f"{'Method':<25} {'Type':<10} {'Response Time':<25} {'논문 결과':<20}")
print("-" * 80)
print(f"{'Elasticsearch BM25':<25} {'Sparse':<10} {np.mean(es_times):.1f}ms ± {np.std(es_times):.1f}ms       {'82ms ± 37ms':<20}")
print(f"{'FAISS L2':<25} {'Dense':<10} {np.mean(faiss_times):.1f}ms ± {np.std(faiss_times):.1f}ms       {'657ms ± 127ms':<20}")
print(f"\n테스트 쿼리: {len(test_queries)}개 (BioASQ 필터링 데이터)")
print(f"FAISS 벡터 수: {NUM_VECTORS:,}개")
print(f"※ 논문은 2.4M 문서 + 서버급 하드웨어 기준, 절대값은 다를 수 있습니다")
print(f"※ 논문 결론: Elasticsearch는 full-text, FAISS는 dense vector에 최적")