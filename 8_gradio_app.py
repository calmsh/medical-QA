"""
STEP 8: Gradio 챗봇 — Medical RAG 서비스 구현
논문 Figure 1의 "Online Phase"를 웹 인터페이스로 구현
"""

import sys
import json
import openai
import os
import gradio as gr

sys.path.append("./rag_system")

from hybrid_retriever import HybridRetriever
from openAI_chat import Chat

client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("Hybrid 리트리버 초기화 중...")
hybrid_retriever = HybridRetriever()
print("Hybrid 리트리버 준비 완료")

chat_full = Chat(question_type=1)
chat_yesno = Chat(question_type=2)


def is_korean(text):
    for char in text:
        if '\uac00' <= char <= '\ud7a3':
            return True
    return False


def translate_to_english(korean_text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Translate the following Korean medical question to English. Return only the translated question, nothing else."},
                {"role": "user", "content": korean_text}
            ],
            max_tokens=200,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return korean_text


def clean_response_text(response_value):
    """
    GPT 응답의 response 필드가 dict/list 등 비문자열일 경우
    사람이 읽을 수 있는 텍스트로 변환
    """
    if isinstance(response_value, str):
        return response_value
    elif isinstance(response_value, dict):
        parts = []
        for key, val in response_value.items():
            label = key.replace('_', ' ')
            if isinstance(val, list):
                items = ", ".join(str(v) for v in val)
                parts.append(f"{label}: {items}")
            else:
                parts.append(f"{label}: {val}")
        return ". ".join(parts) + "."
    elif isinstance(response_value, list):
        return ", ".join(str(v) for v in response_value)
    else:
        return str(response_value)


def generate_english_answer(question_en, retrieved_docs):
    """영어 질문에 대해 영어 답변을 문장 형태로 생성"""
    system_prompt = (
        "You are a scientific medical assistant designed to synthesize responses "
        "from specific medical documents. Only use the information provided in the "
        "documents to answer questions. The first documents should be the most relevant. "
        "Do not use any other information except for the documents provided. "
        "When answering questions, always format your response "
        "as a JSON object with fields for 'response', 'used_PMIDs'. "
        "The 'response' field MUST be a complete, readable English sentence or paragraph. "
        "Do NOT return lists, dictionaries, or structured data in the 'response' field. "
        "Cite all PMIDs your response is based on in the 'used_PMIDs' field. "
        "Please think step-by-step before answering questions and provide the most accurate response possible."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": question_en})

    document_texts = [
        "PMID {}: {} {}".format(doc['PMID'], doc['title'], doc['content'])
        for doc in retrieved_docs.values()
    ]
    documents_message = "\n\n".join(document_texts)
    messages.append({"role": "system", "content": documents_message})

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.0
        )
        response_content = completion.choices[0].message.content
        try:
            response_data = json.loads(response_content)
            return {
                "response": clean_response_text(response_data.get("response", "")),
                "used_PMIDs": response_data.get("used_PMIDs", [])
            }
        except json.JSONDecodeError:
            return {"response": response_content, "used_PMIDs": []}
    except Exception as e:
        return {"response": f"Error: {e}", "used_PMIDs": []}


def translate_to_korean(english_answer):
    """영어 답변을 한국어로 번역 (내용 동일하게 유지)"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Translate the following English medical answer to Korean. "
                 "Translate accurately and completely. Do not add, remove, or summarize any information. "
                 "Return only the translated text, nothing else."},
                {"role": "user", "content": english_answer}
            ],
            max_tokens=500,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"번역 오류: {e}"


def answer_question(question, answer_type, num_docs):
    if not question.strip():
        return "질문을 입력해주세요.", "", ""

    use_korean = is_korean(question)
    if use_korean:
        search_query = translate_to_english(question)
    else:
        search_query = question

    # ── 문서 검색 ──
    try:
        retrieved_docs = json.loads(
            hybrid_retriever.retrieve_docs(search_query, top_n=10, k=int(num_docs))
        )
    except Exception as e:
        return f"검색 오류: {e}", "", ""

    if not retrieved_docs:
        msg = "관련 문서를 찾지 못했습니다."
        return msg, msg, ""

    # ── 영어 답변 생성 ──
    en_result = generate_english_answer(search_query, retrieved_docs)
    en_response = en_result["response"]

    # ── 한국어 답변 생성 (영어 답변 번역) ──
    ko_response = translate_to_korean(en_response)

    # ── 관련 논문 정보 (텍스트 — Textbox용) ──
    docs_lines = []
    doc_list = list(retrieved_docs.values())
    for i, doc in enumerate(doc_list, 1):
        pmid = doc['PMID']
        title = doc['title'][:120]
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        docs_lines.append(f"{i}. {title}")
        docs_lines.append(f"   PMID: {pmid}")
        docs_lines.append(f"   {pubmed_url}")
        docs_lines.append("")
    docs_info = "\n".join(docs_lines)

    return ko_response, en_response, docs_info


# ─── 예시 질문 HTML ───
EXAMPLES_HTML = """
<div style="
    max-height: 360px;
    overflow-y: auto;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: white;
    padding: 0;
">
    <table style="width:100%; border-collapse:collapse; font-size:0.92em;">
        <thead>
            <tr style="background:#f1f5f9; position:sticky; top:0; z-index:1;">
                <th style="padding:10px 16px; text-align:left; border-bottom:2px solid #e2e8f0; color:#334155;">예시 질문</th>
                <th style="padding:10px 12px; text-align:center; border-bottom:2px solid #e2e8f0; color:#334155; width:80px;">유형</th>
                <th style="padding:10px 12px; text-align:center; border-bottom:2px solid #e2e8f0; color:#334155; width:80px;">검색 깊이</th>
            </tr>
        </thead>
        <tbody>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='알츠하이머병은 유전인가요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">알츠하이머병은 유전인가요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='당뇨병의 치료법은 무엇인가요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">당뇨병의 치료법은 무엇인가요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='흡연은 폐암 위험을 증가시키나요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">흡연은 폐암 위험을 증가시키나요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='고혈압의 원인은 무엇인가요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">고혈압의 원인은 무엇인가요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='항생제 내성은 왜 발생하나요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">항생제 내성은 왜 발생하나요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='파킨슨병의 초기 증상은 무엇인가요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">파킨슨병의 초기 증상은 무엇인가요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='비타민D 결핍은 어떤 질환을 유발하나요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">비타민D 결핍은 어떤 질환을 유발하나요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='아스피린은 심장병 예방에 효과가 있나요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">아스피린은 심장병 예방에 효과가 있나요?</td><td style="text-align:center; color:#3b82f6;">Yes/No</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='우울증의 치료 방법에는 무엇이 있나요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">우울증의 치료 방법에는 무엇이 있나요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='코로나19의 주요 증상은 무엇인가요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">코로나19의 주요 증상은 무엇인가요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='유방암의 위험 요인은 무엇인가요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">유방암의 위험 요인은 무엇인가요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='인슐린은 체내에서 어떤 역할을 하나요?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">인슐린은 체내에서 어떤 역할을 하나요?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='Do CpG islands colocalise with transcription start sites?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">Do CpG islands colocalise with transcription start sites?</td><td style="text-align:center; color:#3b82f6;">Yes/No</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='Is Alzheimer\\'s disease hereditary?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">Is Alzheimer's disease hereditary?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='What is the treatment for type 2 diabetes?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">What is the treatment for type 2 diabetes?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">20</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='What are the side effects of chemotherapy?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">What are the side effects of chemotherapy?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='How does immunotherapy work for cancer?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">How does immunotherapy work for cancer?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='What causes Parkinson\\'s disease?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">What causes Parkinson's disease?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='Is aspirin effective for heart disease prevention?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">Is aspirin effective for heart disease prevention?</td><td style="text-align:center; color:#3b82f6;">Yes/No</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='How is tuberculosis diagnosed?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">How is tuberculosis diagnosed?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='What is the role of insulin in the body?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">What is the role of insulin in the body?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='Does metformin help with weight loss?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">Does metformin help with weight loss?</td><td style="text-align:center; color:#3b82f6;">Yes/No</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer; border-bottom:1px solid #f1f5f9;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='What are the symptoms of multiple sclerosis?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">What are the symptoms of multiple sclerosis?</td><td style="text-align:center; color:#6b7280;">상세</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
            <tr style="cursor:pointer;" onmouseover="this.style.background='#f8fafc'" onmouseout="this.style.background='white'" onclick="document.querySelector('#question-box textarea').value='Is CRISPR used in cancer treatment?'; document.querySelector('#question-box textarea').dispatchEvent(new Event('input', {bubbles:true}));">
                <td style="padding:10px 16px;">Is CRISPR used in cancer treatment?</td><td style="text-align:center; color:#3b82f6;">Yes/No</td><td style="text-align:center; color:#6b7280;">50</td>
            </tr>
        </tbody>
    </table>
</div>
"""


# ─── 커스텀 CSS ───
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 50%, #f8faff 100%) !important;
    font-family: 'Pretendard', 'Segoe UI', sans-serif !important;
}

.gr-group, .group {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#input-card {
    background: white !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    border: 1px solid #e2e8f0 !important;
}

#question-box {
    background: transparent !important;
}
#question-box textarea {
    background: white !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 1.05em !important;
    padding: 14px !important;
}
#question-box textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

#submit-btn {
    background: linear-gradient(135deg, #1a56db 0%, #2563eb 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.1em !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
}
#submit-btn:hover {
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* 모든 출력 Textbox 통일 스타일 */
#answer-ko textarea, #answer-en textarea, #docs-box textarea {
    background: white !important;
    font-size: 1.0em !important;
    line-height: 1.7 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

"""


# ─── Gradio 인터페이스 ───
with gr.Blocks(css=custom_css, title="Medical RAG System") as demo:

    # 헤더
    gr.HTML("""
        <div style="
            background: linear-gradient(135deg, #1a56db 0%, #2563eb 50%, #3b82f6 100%);
            border-radius: 16px;
            padding: 32px 40px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(37, 99, 235, 0.3);
        ">
            <h1 style="color:white; font-size:1.7em; font-weight:700; margin:0 0 6px 0;">
                🏥 검색 증강 생성(RAG)을 사용한 효율적인 의학적 질의응답
            </h1>
            <p style="color:rgba(255,255,255,0.9); font-size:1.0em; margin:0;">
                PubMed 기반 의료 문헌 검색과 근거 기반 의료 정보를 제공합니다 &nbsp;|&nbsp; 한국어 / English 지원
            </p>
        </div>
    """)

    # 입력 영역
    with gr.Column(elem_id="input-card"):
        gr.Markdown("### 💬 질문 입력")
        question_input = gr.Textbox(
            label="",
            placeholder="의료 질문을 입력하세요 (예: 알츠하이머는 유전인가요? / Is Alzheimer's disease hereditary?)",
            lines=2,
            elem_id="question-box",
            show_label=False
        )

        with gr.Row():
            answer_type = gr.Radio(
                ["상세 답변", "Yes/No"],
                value="상세 답변",
                label="답변 유형",
                scale=2
            )
            num_docs = gr.Slider(
                minimum=10, maximum=100, value=50, step=10,
                label="후보 문서 수",
                scale=2
            )
            submit_btn = gr.Button(
                "🔍 질문하기",
                variant="primary",
                elem_id="submit-btn",
                scale=1
            )

    gr.HTML("<div style='height:16px'></div>")

    # 답변 영역
    gr.Markdown("### 📋 답변 결과")
    with gr.Row(equal_height=True):
        with gr.Column():
            answer_ko = gr.Textbox(
                label="한국어 답변",
                lines=7,
                elem_id="answer-ko",
                placeholder="질문을 입력하면 PubMed 논문 기반 한국어 답변이 여기에 표시됩니다."
            )
        with gr.Column():
            answer_en = gr.Textbox(
                label="English Answer",
                lines=7,
                elem_id="answer-en",
                placeholder="The English answer based on PubMed articles will be displayed here."
            )

    gr.HTML("<div style='height:12px'></div>")

    # 논문 정보
    docs_output = gr.Textbox(
        label="📄 관련 PubMed 논문 정보",
        lines=8,
        elem_id="docs-box",
        placeholder="질문을 입력하면 검색된 PubMed 논문의 제목, PMID, 링크가 여기에 표시됩니다."
    )

    gr.HTML("<div style='height:16px'></div>")

    # 예시 질문
    gr.Markdown("### 📌 예시 질문 — 클릭하면 질문이 자동 입력됩니다")
    gr.HTML(EXAMPLES_HTML)

    # 하단
    gr.HTML("""
        <div style="text-align:center; color:#64748b; font-size:0.85em; margin-top:20px; padding:12px;">
            Hybrid Retriever (BM25 + MedCPT Cross-Encoder) &nbsp;|&nbsp;
            GPT-3.5-turbo &nbsp;|&nbsp;
            PubMed 2.4M 문서 &nbsp;|&nbsp;
            논문: <em>Efficient and Reproducible Biomedical QA using RAG</em>
        </div>
    """)

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, answer_type, num_docs],
        outputs=[answer_ko, answer_en, docs_output],
        show_progress="minimal"
    )


if __name__ == "__main__":
    print("Gradio 서버 시작...")
    print("브라우저에서 http://localhost:7860 접속")
    demo.launch(server_name="0.0.0.0", server_port=7860)