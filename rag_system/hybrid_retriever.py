"""
Hybrid Retriever — 원본: rag_system/hybrid_retriever.py
논문의 핵심: BM25로 먼저 k개 검색 → MedCPT Cross-Encoder로 리랭킹
변경점: Elasticsearch 연결만 HTTP로 변경
"""
from elasticsearch import Elasticsearch
import os
import json
from medCPT_encoder import MedCPTCrossEncoder

class HybridRetriever:
    def __init__(self):
        # ★ 변경: HTTP 연결
        self.es = Elasticsearch(['http://localhost:9200'], request_timeout=60)
        self.index = "pubmed_index"
        self.reranker = MedCPTCrossEncoder()

    # ─── 이하 원본과 100% 동일 ───
    def rerank_docs(self, query: str, docs: list):
        """Reranks the documents based on their relevance to the query."""
        scores = self.reranker.score([doc['content'] for doc in docs], query)
        reranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return reranked_docs

    def retrieve_docs(self, query: str, top_n: int = 10, k: int = 20):
        """Retrieves documents from Elasticsearch and reranks them."""
        es_query = {
            "size": k,
            "query": {
                "match": {
                    "content": query
                }
            },
            "_source": ["PMID", "title", "content"]
        }
        response = self.es.search(index=self.index, body=es_query)

        docs = [{
            'PMID': hit['_source']['PMID'],
            'title': hit['_source']['title'],
            'content': hit['_source']['content']
        } for hit in response['hits']['hits']]

        reranked_docs = self.rerank_docs(query, docs)

        # only take documents with a score > 0
        reranked_docs = [doc for doc in reranked_docs if doc[1] > 0]

        top_reranked_docs = reranked_docs[:top_n]

        results = {
            f"doc{idx + 1}": {
                'PMID': doc['PMID'],
                'title': doc['title'],
                'content': doc['content'],
                'score': score.item()
            }
            for idx, (doc, score) in enumerate(top_reranked_docs)
        }

        return json.dumps(results, indent=4)