import os
import glob
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="나만의 RAG 챗봇", page_icon="🤖")
st.title("🤖 나만의 DB 기반 RAG 챗봇")

# 1. API 키 확인 (환경 변수에서 로드)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Secrets에 GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# 2. Vector DB 생성 (캐싱 처리로 속도 최적화)
@st.cache_resource
def init_rag_chain():
    # 모든 .md 파일을 찾아 로드
    md_files = glob.glob("*.md")
    if not md_files:
        return None, "저장소에 .md 파일이 없습니다."
    
    docs = []
    for file in md_files:
        loader = TextLoader(file, encoding="utf-8")
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # CPU용 한국어 임베딩 설정 (무료 서버 안정성용)
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Gemini 3.6 Flash 모델 연동
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

# 3. 채팅 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 화면에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("문서에서 답변 찾는 중..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})