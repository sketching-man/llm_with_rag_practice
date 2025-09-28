import json
import time
import uuid
from typing import Any, Dict, Generator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import *
from modular_rag_pipeline import PipelineConfig, build_rag_pipeline, make_handler
from vector_db import ensure_vector_store


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

VECTORSTORE = ensure_vector_store()
RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 4})


# endregion

# region Setup RAG

BASE_CFG = PipelineConfig(
    openai_api_key=OPENAI_API_KEY,
    model=OLLAMA_MODEL,
)
RAG_PIPELINE = build_rag_pipeline(
    cfg=BASE_CFG,
    retriever=RETRIEVER,
    enable_query_rewrite=True,
    enable_translation=False,
)

RAG_HANDLE = make_handler(RAG_PIPELINE)


# endregion

# region FastAPI app

app = FastAPI(title="OpenAI-compatible RAG Server (LangChain)")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    # Authorization header is accepted but not enforced
    debug_flag = True  # Turn off if not debugging
    question = _extract_user_question(req.messages)
    result = RAG_HANDLE(query=question, debug=debug_flag)
    answer = result["answer"]

    if req.stream:
        return StreamingResponse(
            _generate_streaming(answer, req.model),
            media_type="text/event-stream",
        )
    payload = _openai_like_payload(answer, req.model)
    if not req.stream and debug_flag:
        payload["x_rag_debug"] = {
            "query_rewritten": result.get("query_rewritten"),
            "sources": result.get("sources"),
            "scores": (result.get("debug") or {}).get("scores"),
            "previews": (result.get("debug") or {}).get("previews"),
        }

    return JSONResponse(content=payload)


@app.get("/v1/models")
async def list_models():
    # Minimal model listing for OpenAI compatibility
    # model_id = OPENAI_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL
    model_id = "my_model"
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

# TODO: 스트리밍 응답 점검
