import os
import glob
import uuid
import streamlit as st
from supabase import create_client, Client

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="게임 스토리 AI")

# 1. API 키 및 DB 연동
api_key = os.environ.get("GEMINI_API_KEY")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not api_key:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

supabase: Client = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)

# 2. Vector DB 생성 (Gemini 최신 임베딩 모델)
@st.cache_resource
def init_rag_chain():
    md_files = glob.glob("**/*.md", recursive=True)
    if not md_files:
        return None, "저장소에 .md 파일이 없습니다."
    
    docs = []
    for file in md_files:
        loader = TextLoader(file, encoding="utf-8")
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.1)

    prompt_template = """다음 제공된 참고 문서 내용만을 바탕으로 질문에 정확하게 답변해 주세요.
만약 제공된 문서 내용에서 질문에 대한 근거나 출처를 찾을 수 없거나 불확실한 경우, 지어내지 말고 반드시 "제공된 문서에서 출처를 확인할 수 없거나 불확실한 정보입니다."라고 답변하세요.

# 참고 문서:
{context}

# 질문:
{question}

# 답변:"""

    prompt = PromptTemplate.from_template(prompt_template)
    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, None

rag_chain, error_msg = init_rag_chain()
if error_msg:
    st.warning(error_msg)
    st.stop()

# 3. 사이드바 - 대화 목록 관리 (ChatGPT 방식)
st.sidebar.title("대화 목록")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if st.sidebar.button("➕ 새 대화 시작"):
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

# 저장된 세션 목록 불러오기
sessions = []
if supabase:
    try:
        res = supabase.table("chat_history").select("session_id, content").eq("role", "user").order("created_at", desc=True).execute()
        seen = set()
        for row in res.data:
            s_id = row["session_id"]
            if s_id not in seen:
                seen.add(s_id)
                preview = row["content"][:15] + "..." if len(row["content"]) > 15 else row["content"]
                sessions.append((s_id, preview))
    except Exception:
        pass

if sessions:
    st.sidebar.write("---")
    session_dict = {f"{preview}": s_id for s_id, preview in sessions}
    selected_label = st.sidebar.radio("이전 대화 선택", list(session_dict.keys()))
    if selected_label:
        st.session_state.session_id = session_dict[selected_label]

# 선택된 세션의 대화 내역 DB에서 로드
messages = []
if supabase:
    try:
        res = supabase.table("chat_history").select("role, content").eq("session_id", st.session_state.session_id).order("created_at", asc=True).execute()
        messages = res.data
    except Exception:
        messages = []

# 메인 UI
st.title("게임 스토리 AI")

# 기존 메시지 화면 출력
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 질문 처리
if user_input := st.chat_input("질문을 입력하세요..."):
    # 1. 화면 출력
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. DB 저장 (사용자 질문)
    if supabase:
        supabase.table("chat_history").insert({
            "session_id": st.session_state.session_id,
            "role": "user",
            "content": user_input
        }).execute()

    # 3. AI 답변 생성 및 화면 출력
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)

    # 4. DB 저장 (AI 답변)
    if supabase:
        supabase.table("chat_history").insert({
            "session_id": st.session_state.session_id,
            "role": "assistant",
            "content": response
        }).execute()
        
    st.rerun()
