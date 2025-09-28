from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter

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
    docs: list[Document] = []
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

    chunks = _split_document(docs)
    vs = FAISS.from_documents(chunks, embedding)
    vs.save_local(FAISS_DIR)
    return vs


def _split_document(docs: list[Document]):
    chunks: list[Document] = []

    def _base_meta(doc: Document) -> dict:
        m = dict(doc.metadata or {})
        # 파일 경로가 있다면 보존
        if "source" in m:
            m["source"] = m["source"]
        return m

    if CHUNK_STRATEGY == "markdown" and MarkdownHeaderTextSplitter is not None:
        # 1) 헤더 단위 1차 분해
        md_splits: list[Document] = []
        for d in docs:
            src = os.path.basename(str((d.metadata or {}).get("source", "")))
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
            )
            sections = splitter.split_text(d.page_content or "")
            # sections는 Document가 아니므로 다시 래핑
            for sec in sections:
                md_splits.append(
                    Document(
                        page_content=sec.page_content,
                        metadata={**_base_meta(d), "source": (d.metadata or {}).get("source"), **sec.metadata},
                    )
                )
        # 2) 섹션이 아직 길면 재귀 분해
        rec = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        for i, d in enumerate(md_splits):
            sub = rec.split_documents([d])
            for j, c in enumerate(sub):
                c.metadata.update({"chunk": j, "section_path": _section_path(c.metadata)})
            chunks.extend(sub)

    elif CHUNK_STRATEGY == "token":
        # 토큰 기반(모델 토크나이저에 근사). 길이에 민감한 모델 쓸 때 유용
        t = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for d in docs:
            sub = t.split_documents([d])
            for j, c in enumerate(sub):
                c.metadata.update({**_base_meta(d), "chunk": j})
            chunks.extend(sub)
    else:
        # 기본: 재귀 문자 기반
        rec = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        for d in docs:
            sub = rec.split_documents([d])
            for j, c in enumerate(sub):
                c.metadata.update({**_base_meta(d), "chunk": j})
            chunks.extend(sub)

    return chunks


def _section_path(meta: dict) -> str:
    """markdown 분해 시 섹션 경로 문자열(h1 > h2 > h3 ...)"""
    keys = ["h1", "h2", "h3", "h4", "h5", "h6"]
    parts = [str(meta[k]) for k in keys if k in meta and meta[k]]
    return " > ".join(parts)
