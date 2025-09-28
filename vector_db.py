from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import *


def ensure_vector_store() -> FAISS:
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    embedding = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    # 1. 기존 인덱스가 있으면 로드
    index_file = os.path.join(FAISS_DIR, "index.faiss")
    store_file = os.path.join(FAISS_DIR, "index.pkl")
    if os.path.exists(index_file) and os.path.exists(store_file):
        return FAISS.load_local(
            FAISS_DIR,
            embeddings=embedding,
            allow_dangerous_deserialization=True,  # 로컬 신뢰 환경에서만 사용
        )

    # 2. 없으면 문서 수집 후 새로 생성
    docs = []
    if os.path.isdir(DOCS_DIR):
        loader_txt = DirectoryLoader(
            DOCS_DIR, glob="**/*.txt",
            loader_cls=TextLoader, loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
        )
        docs += loader_txt.load()

        loader_md = DirectoryLoader(
            DOCS_DIR, glob="**/*.md",
            loader_cls=TextLoader, loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
        )
        docs += loader_md.load()

        # loader_pdf = DirectoryLoader(
        #     DOCS_DIR, glob="**/*.pdf",
        #     loader_cls=PyPDFLoader,
        #     show_progress=True,
        # )
        # docs += loader_pdf.load()

    if len(docs) == 0:
        docs = [Document(page_content="Empty KB. Add files to ./data/docs and restart.")]

    vs = FAISS.from_documents(docs, embedding)
    vs.save_local(FAISS_DIR)
    return vs