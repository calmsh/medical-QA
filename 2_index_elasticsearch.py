"""
STEP 2: Elasticsearch 인덱싱
논문 Section III-B "Indexing and Query Time" 대응
원본 파일: information_retrieval/elastic_container/ingest_data.py

변경점 (원본 대비):
- Docker 내부 경로 → Windows 로컬 경로
- HTTPS + 인증 → HTTP (보안 비활성화 상태)
- 나머지 로직은 원본 그대로
"""

import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

# ─── Elasticsearch 연결 (원본: HTTPS + 인증서, 수정: HTTP) ───
# 원본 코드 (ingest_data.py):
#   es = Elasticsearch(
#       hosts=[{"host": "localhost", "port": 9200, "scheme": "https"}],
#       ca_certs="/home/rag/.crt/http_ca.crt",
#       basic_auth=("elastic", password),
#   )
# 수정: 보안 비활성화 상태이므로 HTTP로 연결
es = Elasticsearch(['http://localhost:9200'])

# 연결 확인
if es.ping():
    print("Elasticsearch 연결 성공!")
else:
    print("Elasticsearch 연결 실패! elasticsearch.bat이 실행 중인지 확인하세요.")
    exit(1)

# ─── 인덱스 생성 (원본 ingest_data.py 그대로) ───
index_name = "pubmed_index"

# 기존 인덱스 삭제
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"기존 인덱스 '{index_name}' 삭제 완료")

# 매핑 정의 (원본 그대로 — BM25 + stopword 제거)
mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "standard",
                    "stopwords": "_english_"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "default",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            }
        }
    }
}

es.indices.create(index=index_name, body=mapping)
print(f"인덱스 '{index_name}' 생성 완료")

# ─── 데이터 벌크 인덱싱 (원본 ingest_data.py 로직 그대로) ───
source_file = Path('./data/pubmed_subset.jsonl')

start_time = time.time()
actions = []
total_docs = 0

with open(source_file, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="인덱싱 중"):
        try:
            doc = json.loads(line)
            action = {
                "_index": index_name,
                "_source": doc
            }
            actions.append(action)

            # 원본과 동일: 200건씩 벌크 인덱싱
            if len(actions) == 200:
                helpers.bulk(es, actions)
                total_docs += len(actions)
                actions = []
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 에러: {e}")

# 남은 문서 인덱싱
if actions:
    helpers.bulk(es, actions)
    total_docs += len(actions)

indexing_time = time.time() - start_time

# 인덱스 새로고침 (검색 가능하도록)
es.indices.refresh(index=index_name)

# ─── 결과 출력 ───
count_result = es.count(index=index_name)
print(f"\n인덱싱 완료!")
print(f"인덱스 내 문서 수: {count_result['count']}")
print(f"인덱싱 소요 시간: {indexing_time:.2f}초")
print(f"※ 논문 Table I: Elasticsearch BM25 인덱싱 = 156분 (2.4M 문서 기준)")