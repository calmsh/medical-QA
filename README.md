# 🏥 Medical RAG System

> **PubMed 기반 검색 증강 생성(RAG)을 활용한 의학적 질의응답 시스템**
>
> 논문: *Efficient and Reproducible Biomedical Question Answering using Retrieval Augmented Generation*  
> (IEEE SDS 2025 — Linus Stuhlmann, Michael Saxer, Jonathan Fürst)

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [전체 실행 흐름](#-전체-실행-흐름)
- [파일 구조](#-파일-구조)
- [스텝별 상세 설명](#-스텝별-상세-설명)
- [RAG 시스템 핵심 모듈](#-rag-시스템-핵심-모듈-rag_system)
- [환경 설정](#-환경-설정)
- [실험 환경](#-실험-환경)
- [논문 인용](#-논문-인용)

---

## 🔍 프로젝트 개요

이 프로젝트는 PubMed 2.4M 문서를 기반으로 한 **의학 질의응답 RAG 시스템**을 구현합니다.

- **검색(Retrieval)**: BM25(Elasticsearch) + MedCPT Cross-Encoder 리랭킹의 하이브리드 방식  
- **생성(Generation)**: GPT-3.5-turbo를 이용한 한국어/영어 답변 생성  
- **평가(Evaluation)**: BioASQ 데이터셋을 활용한 Recall, Precision, Accuracy, F1 측정  
- **서비스(App)**: Gradio 기반 웹 인터페이스 제공

---

## 🔄 전체 실행 흐름

```
[STEP 1]               [STEP 2]                [STEP 3]
PubMed 데이터          Elasticsearch            BioASQ 데이터
다운로드       ──►     인덱싱          ──►      전처리/필터링
(HuggingFace)          (BM25 색인)              (평가 데이터 준비)
     │                      │                        │
     └──────────────────────┴────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          [STEP 4]       [STEP 5]       [STEP 6]
          데이터 저장소   리트리버 성능   엔드투엔드
          비교           비교           RAG 평가
          (Table I)      (Table II)     (Table III)
                              │
                         [STEP 7]
                         Retrieval Depth
                         분석 (Table IV)
                              │
                         [STEP 8]
                         Gradio 챗봇
                         웹 서비스
```

---

## 📁 파일 구조

```
medical_rag/
│
├── 1_download_data.py          # STEP 1: PubMed 데이터 다운로드
├── 2_index_elasticsearch.py    # STEP 2: Elasticsearch BM25 인덱싱
├── 3_preprocess_bioasq.py      # STEP 3: BioASQ 평가 데이터 전처리
├── 4_compare_datastores.py     # STEP 4: ES vs FAISS 응답 시간 비교
├── 5_compare_retrievers.py     # STEP 5: BM25 vs Hybrid 리트리버 성능 비교
├── 6_evaluate_rag.py           # STEP 6: GPT + 리트리버 엔드투엔드 평가
├── 7_retrieval_depth.py        # STEP 7: 검색 깊이(k) 파라미터 최적화
├── 8_gradio_app.py             # STEP 8: Gradio 웹 챗봇 서비스
│
├── rag_system(논문 원본)/       # RAG 핵심 모듈
│   ├── bm25_retriever.py       # BM25 검색기 (Elasticsearch)
│   ├── hybrid_retriever.py     # Hybrid 검색기 (BM25 + MedCPT 리랭킹)
│   ├── medCPT_encoder.py       # MedCPT 임베딩 인코더
│   ├── medCPT_retriever.py     # MedCPT Dense 검색기
│   ├── bioBERT_encoder.py      # BioBERT 임베딩 인코더
│   ├── bioBERT_retriever.py    # BioBERT Dense 검색기
│   ├── openAI_chat.py          # GPT-3.5-turbo 답변 생성
│   ├── med_rag.py              # RAG 파이프라인 통합
│   └── pipeline.ipynb          # 전체 파이프라인 노트북
│
├── data/                       # 데이터 저장 폴더 (자동 생성)
│   ├── pubmed_subset.jsonl     # STEP 1 결과: PubMed 문서
│   └── bioasq_filtered/        # STEP 3 결과: 필터링된 BioASQ 질문
│       ├── all_filtered_questions.json
│       ├── yesno_questions.json
│       ├── factoid_questions.json
│       ├── list_questions.json
│       └── summary_questions.json
│
├── elasticsearch/              # Elasticsearch 설치 폴더
├── requirements.txt            # Python 패키지 목록
└── README.md
```

---

## 📌 스텝별 상세 설명

### STEP 1 — PubMed 데이터 다운로드 (`1_download_data.py`)

HuggingFace에서 PubMed 2.4M 문서 서브셋을 다운로드하여 JSONL 형식으로 저장합니다.

- **출처**: [`slinusc/PubMedAbstractsSubset`](https://huggingface.co/datasets/slinusc/PubMedAbstractsSubset)
- **출력**: `./data/pubmed_subset.jsonl`
- **데이터 형식**: `{"PMID": "...", "title": "...", "content": "..."}`

```bash
python 1_download_data.py
```

---

### STEP 2 — Elasticsearch 인덱싱 (`2_index_elasticsearch.py`)

다운로드한 PubMed 문서를 Elasticsearch에 BM25 인덱스로 저장합니다.

- **전제 조건**: Elasticsearch 실행 필요 (`elasticsearch/bin/elasticsearch.bat`)
- **인덱스명**: `pubmed_index`
- **분석기**: Standard Analyzer + 영어 불용어 제거

```bash
python 2_index_elasticsearch.py
```

> 📌 논문 Table I: 2.4M 문서 기준 인덱싱 소요 시간 = **156분**

---

### STEP 3 — BioASQ 전처리 (`3_preprocess_bioasq.py`)

BioASQ 질의응답 데이터에서 인덱스에 존재하는 PMID를 가진 질문만 필터링합니다.

- **입력**: `./data/training13b.json` (BioASQ 원본)
- **출력**: `./data/bioasq_filtered/` (유형별 JSON 파일)
- **필터 조건**: ground truth PMID가 1개 이상 인덱스에 존재하는 질문

```bash
python 3_preprocess_bioasq.py
```

> 📌 논문 결과: 전체 30,212개 질문 중 **723개** 필터링 (3.04%)

---

### STEP 4 — 데이터 저장소 비교 (`4_compare_datastores.py`)

Elasticsearch(BM25)와 FAISS(Dense Vector)의 응답 시간을 비교합니다.

| 저장소 | 유형 | 논문 응답 시간 |
|--------|------|----------------|
| Elasticsearch BM25 | Sparse | **82ms ± 37ms** |
| FAISS L2 | Dense | **657ms ± 127ms** |

```bash
python 4_compare_datastores.py
```

---

### STEP 5 — 리트리버 성능 비교 (`5_compare_retrievers.py`)

BM25와 Hybrid 리트리버의 Recall/Precision을 BioASQ 데이터로 비교합니다.

| 리트리버 | Recall | Precision |
|----------|--------|-----------|
| BM25 | 0.537 | 0.322 |
| **Hybrid (BM25 + MedCPT)** | **0.567** | **0.319** |

```bash
python 5_compare_retrievers.py
```

---

### STEP 6 — 엔드투엔드 RAG 평가 (`6_evaluate_rag.py`)

BioASQ yesno 질문으로 GPT-3.5-turbo + 리트리버 조합의 정답률을 평가합니다.

| RAG 구성 | Accuracy | F1 |
|----------|----------|----|
| GPT-3.5 + BM25 | 0.72 | 0.74 |
| **GPT-3.5 + Hybrid** | **0.86** | **0.86** |

```bash
python 6_evaluate_rag.py
```

---

### STEP 7 — Retrieval Depth 분석 (`7_retrieval_depth.py`)

BM25로 검색할 후보 문서 수(k)를 변경하며 최적값을 탐색합니다.

| k (후보 문서 수) | Accuracy |
|------------------|----------|
| 20 | 0.89 |
| **50** | **0.90** ← 최적 |
| 100 | 0.87 |

```bash
python 7_retrieval_depth.py
```

> 📌 k=100은 오히려 성능 하락 — 관련성 낮은 문서가 노이즈로 작용

---

### STEP 8 — Gradio 챗봇 앱 (`8_gradio_app.py`)

Hybrid Retriever + GPT-3.5-turbo 기반 의학 질의응답 웹 서비스입니다.

- **한국어 / English** 질문 모두 지원 (한국어 입력 시 자동 영어 번역 후 검색)
- **검색 결과**: 관련 PubMed 논문 제목, PMID, 링크 표시
- **접속 주소**: `http://localhost:7860`

```bash
python 8_gradio_app.py
```

---

## ⚙️ 환경 설정

### 1. 가상환경 생성 및 패키지 설치

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. OpenAI API 키 설정

```bash
# Windows
set OPENAI_API_KEY=sk-...

# PowerShell
$env:OPENAI_API_KEY="sk-..."
```

### 3. Elasticsearch 실행

```bash
elasticsearch\bin\elasticsearch.bat
```

### 4. 스텝 순서대로 실행

```bash
python 1_download_data.py
python 2_index_elasticsearch.py
python 3_preprocess_bioasq.py
# 이후 4~7은 평가 목적, 8은 서비스 실행
python 8_gradio_app.py
```

---

## 📄 논문 인용

```bibtex
@INPROCEEDINGS{11081505,
  author={Stuhlmann, Linus and Saxer, Michael Alexander and Fürst, Jonathan},
  booktitle={2025 IEEE Swiss Conference on Data Science (SDS)},
  title={Efficient and Reproducible Biomedical Question Answering Using Retrieval Augmented Generation},
  year={2025},
  pages={154-157},
  doi={10.1109/SDS66131.2025.00029}
}
```

[📖 arXiv에서 논문 읽기](https://arxiv.org/abs/2505.07917)

---

## 📦 데이터셋

- **PubMed 문서**: [slinusc/PubMedAbstractsSubset](https://huggingface.co/datasets/slinusc/PubMedAbstractsSubset) (2.4M 문서)
- **사전 계산 임베딩**: [slinusc/PubMedAbstractsSubsetEmbedded](https://huggingface.co/datasets/slinusc/PubMedAbstractsSubsetEmbedded) (MedCPT 벡터)
- **BioASQ 평가 데이터**: [BioASQ 공식 사이트](http://www.bioasq.org/) 에서 `training13b.json` 다운로드 필요

---
