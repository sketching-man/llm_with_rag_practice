import json
import os
import time
from typing import Any, Dict, Generator, Optional
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

from pydantic import BaseModel, SecretStr

# region Config
DOCS_DIR = os.getenv("DOCS_DIR", "./data/docs")
FAISS_DIR = os.getenv("FAISS_DIR", "./data/faiss_index")

# LLM provider selection:
#   LLM_PROVIDER=ollama (default) or openai
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI/vLLM-compatible settings
OPENAI_API_KEY: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", "not-needed-for-local"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embeddings model
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# endregion

# region Data models (OpenAI Schema)
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


# endregion

# region Build/Load vector store (FAISS)
def _ensure_vector_store() -> FAISS:
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


VECTORSTORE = _ensure_vector_store()
RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 4})


# endregion

# region LLM factory
def make_llm(temperature: float):
    if LLM_PROVIDER == "openai":
        # Works with OpenAI AND any OpenAI-compatible base_url (e.g., vLLM)
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,  # None -> api.openai.com
            temperature=temperature,
        )

    # default: Ollama
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=temperature,
    )


# endregion

# region RAG chain
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the following retrieved context to answer the user.
    If the answer is not in the context, say you don't know concisely.
    
    # Context
    {context}
    
    # Instruction
    Answer in the same language as the user's last message. Be concise, but include key details when helpful.
    
    # User
    {question}"""
)


def build_chain(temperature: float = 0.2):
    llm = make_llm(temperature)

    def _format_docs(docs: list[Document]) -> str:
        parts = []
        for d in docs:
            meta = d.metadata or {}
            src = meta.get("source")
            page = meta.get("page")
            tag = f"[source={src} page={page}]" if src else ""
            parts.append(f"{tag}\n{d.page_content}")
        return "\n\n---\n\n".join(parts)

    chain = (
            {"context": RETRIEVER | _format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT_TEMPLATE | llm | StrOutputParser()
    )
    return chain


# endregion

# region FastAPI app
app = FastAPI(title="OpenAI-compatible RAG Server (LangChain)")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    # Authorization header is accepted but not enforced
    question = _extract_user_question(req.messages)
    chain = build_chain(temperature=req.temperature or 0.2)
    answer = chain.invoke(question)

    if req.stream:
        return StreamingResponse(
            _generate_streaming(answer, req.model),
            media_type="text/event-stream",
        )
    payload = _openai_like_payload(answer, req.model)
    return JSONResponse(content=payload)


@app.get("/v1/models")
async def list_models():
    # Minimal model listing for OpenAI compatibility
    model_id = OPENAI_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL
    now = int(time.time())

    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": now,
                "owned_by": "owner",
            }
        ],
    }


def _extract_user_question(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content

    # fallback: join all
    return "\n".join([m.content for m in messages])


def _generate_streaming(answer: str, model: str) -> Generator[bytes, None, None]:
    # OpenAI streaming format with "data: {json}\n\n" chunks
    now = int(time.time())
    head = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n".encode("utf-8")

    for chunk in answer:
        chunk_payload = {
            "id": head["id"],
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n".encode("utf-8")
        # time.sleep(0.01)  # simulate slight delay if needed

    tail = {
        "id": head["id"],
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def _openai_like_payload(answer: str, model: str) -> Dict[str, Any]:
    now = int(time.time())

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }
# endregion
