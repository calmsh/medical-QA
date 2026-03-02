"""
STEP 3: BioASQ 데이터 전처리 (논문 평가 데이터 준비)
논문 원본: evaluation/evaluation_QA_system/dataset_filter/filter_data.ipynb

워크플로 (원본 그대로):
  1. training13b.json에서 질문별 ground truth PMID 추출
  2. Elasticsearch 인덱스에 존재하는 PMID 확인
  3. ground truth PMID가 1개 이상 인덱스에 있는 질문만 필터링
  4. 질문 유형별(yesno, factoid, list, summary) 분리 저장

※ 논문 결과: 전체 30,212개 질문 중 723개가 ≥1 매칭 (3.04%)
※ 이 스크립트는 step5~7에서 사용할 평가 데이터를 생성합니다
"""

import json
import os
import re
from tqdm import tqdm
from elasticsearch import Elasticsearch

# ─── 설정 ───
BIOASQ_PATH = "./data/training13b.json"        # BioASQ 원본 데이터
OUTPUT_DIR = "./data/bioasq_filtered"           # 필터링된 데이터 저장 경로
MIN_MATCH = 1                                    # 최소 매칭 PMID 수 (논문: ≥1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. Elasticsearch 연결 ───
es = Elasticsearch(['http://localhost:9200'])
if not es.ping():
    print("Elasticsearch 연결 실패! elasticsearch.bat이 실행 중인지 확인하세요.")
    exit(1)

total_docs = es.count(index='pubmed_index')['count']
print(f"Elasticsearch 인덱스 문서 수: {total_docs:,}개")

# ─── 2. BioASQ 데이터 로드 ───
print(f"\nBioASQ 데이터 로드 중: {BIOASQ_PATH}")
with open(BIOASQ_PATH, 'r', encoding='utf-8') as f:
    bioasq_data = json.load(f)

questions = bioasq_data.get('questions', [])
print(f"전체 질문 수: {len(questions):,}개")

# 질문 유형별 통계
type_counts = {}
for q in questions:
    qtype = q.get('type', 'unknown')
    type_counts[qtype] = type_counts.get(qtype, 0) + 1
print(f"질문 유형별 분포: {type_counts}")

# ─── 3. 모든 ground truth PMID 수집 (원본 filter_data.ipynb Cell 1) ───
print("\n모든 ground truth PMID 수집 중...")
all_gt_pmids = set()
for q in tqdm(questions, desc="PMID 추출"):
    for url in q.get('documents', []):
        match = re.search(r'pubmed/(\d+)', url)
        if match:
            all_gt_pmids.add(match.group(1))

print(f"고유 ground truth PMID 수: {len(all_gt_pmids):,}개")

# ─── 4. Elasticsearch 인덱스에 존재하는 PMID 확인 (원본 filter_data.ipynb Cell 2) ───
print("\nElasticsearch 인덱스에서 PMID 존재 여부 확인 중...")
print("(이 과정은 데이터 규모에 따라 시간이 걸릴 수 있습니다)")

matched_pmids = set()
pmid_list = list(all_gt_pmids)
batch_size = 500  # 한 번에 검색할 PMID 수

for i in tqdm(range(0, len(pmid_list), batch_size), desc="PMID 매칭"):
    batch = pmid_list[i:i + batch_size]
    # terms 쿼리로 일괄 확인 (원본보다 효율적인 방식)
    try:
        result = es.search(
            index='pubmed_index',
            body={
                "size": 0,
                "query": {
                    "terms": {"PMID": batch}
                },
                "aggs": {
                    "found_pmids": {
                        "terms": {
                            "field": "PMID",
                            "size": len(batch)
                        }
                    }
                }
            }
        )
        for bucket in result['aggregations']['found_pmids']['buckets']:
            matched_pmids.add(str(bucket['key']))
    except Exception as e:
        # terms 쿼리 실패 시 개별 검색으로 폴백
        for pmid in batch:
            try:
                count = es.count(
                    index='pubmed_index',
                    body={"query": {"term": {"PMID": pmid}}}
                )['count']
                if count > 0:
                    matched_pmids.add(pmid)
            except:
                pass

# ─── 원본 filter_data.ipynb의 비율 계산 ───
match_percentage = (len(matched_pmids) / len(all_gt_pmids) * 100) if all_gt_pmids else 0
print(f"\n인덱스에 존재하는 PMID: {len(matched_pmids):,}개 / {len(all_gt_pmids):,}개")
print(f"매칭 비율: {match_percentage:.2f}%")
print(f"※ 논문 결과: 3.04% (2.4M 서브셋 기준)")

# ─── 5. 질문 필터링 (원본 filter_data.ipynb Cell 3) ───
# "at least one pubmed id as answer which is present in our dataset"
print(f"\n질문 필터링 중 (최소 {MIN_MATCH}개 PMID 매칭)...")

selected_questions = []
for q in questions:
    doc_urls = q.get('documents', [])
    pmids_in_question = []
    for url in doc_urls:
        match = re.search(r'pubmed/(\d+)', url)
        if match:
            pmids_in_question.append(match.group(1))

    # 매칭 PMID 수 확인
    match_count = sum(1 for pid in pmids_in_question if pid in matched_pmids)

    if match_count >= MIN_MATCH:
        # documents 필드에서 인덱스에 있는 PMID의 URL만 남기기
        # (원본 filter_data.ipynb의 filter_documents_in_json 함수)
        filtered_docs = [
            url for url in doc_urls
            if re.search(r'pubmed/(\d+)', url) and
               re.search(r'pubmed/(\d+)', url).group(1) in matched_pmids
        ]
        q_copy = q.copy()
        q_copy['documents'] = filtered_docs
        selected_questions.append(q_copy)

print(f"필터링된 질문 수: {len(selected_questions)}개 / {len(questions):,}개")
print(f"※ 논문 결과: 723개 (≥1 매칭 기준)")

# ─── 6. 질문 유형별 분리 저장 (원본 filter_data.ipynb Cell 마지막) ───
questions_by_type = {"yesno": [], "list": [], "summary": [], "factoid": []}

for q in selected_questions:
    qtype = q.get('type', '')
    if qtype in questions_by_type:
        questions_by_type[qtype].append(q)

# 전체 저장
all_path = os.path.join(OUTPUT_DIR, "all_filtered_questions.json")
with open(all_path, 'w', encoding='utf-8') as f:
    json.dump({"questions": selected_questions}, f, indent=2, ensure_ascii=False)

# 유형별 저장
print(f"\n유형별 질문 수:")
for qtype, qs in questions_by_type.items():
    filepath = os.path.join(OUTPUT_DIR, f"{qtype}_questions.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({"questions": qs}, f, indent=2, ensure_ascii=False)
    print(f"  {qtype}: {len(qs)}개 → {filepath}")

# ─── 7. 요약 ───
print("\n" + "=" * 60)
print("BioASQ 전처리 완료 — 논문 filter_data.ipynb 재현")
print("=" * 60)
print(f"입력: {BIOASQ_PATH} ({len(questions):,}개 질문)")
print(f"출력: {OUTPUT_DIR}/ ({len(selected_questions)}개 필터링된 질문)")
print(f"  - all_filtered_questions.json (전체)")
print(f"  - yesno_questions.json ({len(questions_by_type['yesno'])}개) ← Table II, III, IV 평가용")
print(f"  - factoid_questions.json ({len(questions_by_type['factoid'])}개)")
print(f"  - list_questions.json ({len(questions_by_type['list'])}개)")
print(f"  - summary_questions.json ({len(questions_by_type['summary'])}개)")