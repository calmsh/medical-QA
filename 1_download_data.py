"""
STEP 1: PubMed 데이터 다운로드
논문 Section III-A "Datasets" 대응
- PubMed 2.4M 문서 서브셋 (10% of 24M)
- 각 문서: PMID, title, abstract(content)
"""

from datasets import load_dataset
import json
import os

# ─── 설정 ───
# 실습용으로 데이터 일부만 사용 (전체: 2.4M, 실습용: 10,000건 권장)
# 전체를 쓰려면 NUM_SAMPLES = None 으로 변경
NUM_SAMPLES = None
OUTPUT_DIR = "./data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── HuggingFace에서 데이터 다운로드 ───
print("HuggingFace에서 PubMed 데이터셋 다운로드 중...")
print("(처음 실행 시 시간이 걸릴 수 있습니다)")

dataset = load_dataset("slinusc/PubMedAbstractsSubset", split="train")

if NUM_SAMPLES:
    dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
    print(f"실습용으로 {NUM_SAMPLES}건만 사용합니다.")

print(f"총 {len(dataset)}건의 문서를 로드했습니다.")
print(f"데이터 컬럼: {dataset.column_names}")
print(f"예시 데이터:")
print(json.dumps(dataset[0], indent=2, ensure_ascii=False)[:500])

# ─── JSONL 파일로 저장 ───
output_path = os.path.join(OUTPUT_DIR, "pubmed_subset.jsonl")

with open(output_path, "w", encoding="utf-8") as f:
    for item in dataset:
        # 원본 GitHub ingest_data.py의 형식에 맞춤
        doc = {
            "PMID": str(item.get("PMID", item.get("pmid", ""))),
            "title": item.get("title", ""),
            "content": item.get("content", item.get("abstract", ""))
        }
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"\n저장 완료: {output_path}")
print(f"파일 크기: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")