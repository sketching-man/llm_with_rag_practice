from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import *


# region Config + simple prompt store

@dataclass
class PipelineConfig:
    openai_api_key: SecretStr
    model: str = OLLAMA_MODEL
    temperature: float = 0.2
    max_context_chars: int = 32000  # approximately few thousand tokens
    target_language: str = "ko"  # used when enable_translation=True


# endregion

# region Core

@dataclass
class ModuleData:
    query: str = ""
    query_refined: str = ""
    documents: List[Document] = field(default_factory=list)
    answer: str = ""
    sources: List[str] = field(default_factory=list)


class BaseModule:
    def execute(self, data: ModuleData) -> ModuleData:  # pragma: no cover
        raise NotImplementedError


class Pipeline:
    def __init__(self):
        self._modules: List[BaseModule] = []

    def add(self, module: BaseModule) -> "Pipeline":
        self._modules.append(module)
        return self

    def execute(self, query: str) -> ModuleData:
        data = ModuleData(query=query, query_refined=query)
        for m in self._modules:
            data = m.execute(data)
        return data


# endregion

# region Utilities

def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


def _get_llm(temperature: Optional[float] = None) -> BaseChatModel:
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

# region Modules
class QueryRewriteModule(BaseModule):
    """Optional: rewrite query for retrieval."""

    def __init__(self, llm: BaseChatModel):
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=(
                    "Rewrite the user's query to maximize retrieval quality.\n"
                    "Return only the rewritten query.\n\nUser query: {query}"
                ),
                input_variables=["query"],
            ),
        )

    def execute(self, data: ModuleData) -> ModuleData:
        out = self.chain({"query": data.query})["text"]
        data.query_refined = out.strip() or data.query
        return data


class TranslatorModule(BaseModule):
    """Optional: translate to a fixed target language before retrieval."""

    def __init__(self, llm: BaseChatModel, target_language: str = "ko"):
        self.target_language = target_language
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"Translate the user's query to {target_language}. Return only the translation."),
            ("human", "{text}"),
        ])
        self.llm = llm

    def execute(self, data: ModuleData) -> ModuleData:
        # bypass if already Korean and target is Korean
        if self.target_language.lower().startswith("ko") and any(
                "\uac00" <= ch <= "\ud7a3" for ch in data.query_refined):
            return data
        msgs = self.prompt.format_messages(text=data.query_refined)
        resp = self.llm.invoke(msgs)
        data.query_refined = _to_text(resp.content) or data.query_refined
        return data


class RetrieverModule(BaseModule):
    def __init__(self, retriever):
        if retriever is None:
            raise ValueError("A VectorDB retriever must be provided.")
        self.retriever = retriever

    def execute(self, data: ModuleData) -> ModuleData:
        docs = self.retriever.get_relevant_documents(data.query_refined)
        # store sources for client
        sources: List[str] = []
        for d in docs:
            src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
            if src:
                sources.append(os.path.basename(str(src)))
        data.documents = docs
        data.sources = sources
        return data


class AnswerModule(BaseModule):
    """Single-pass synthesis strictly from retrieved context."""

    def __init__(self, llm: BaseChatModel, max_context_chars: int):
        self.llm = llm
        self.max_context_chars = max_context_chars
        self.prompt = PromptTemplate(
            template=(
                "You are a precise assistant. Answer ONLY with information from the provided context.\n"
                "If the answer is not in the context, reply: '정보 없음'.\n"
                "Keep the answer concise and in Korean.\n\n"
                "[Context]\n{context}\n\n[Question]\n{question}"
            ),
            input_variables=["context", "question"],
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    @staticmethod
    def _build_context(docs: List[Document], limit: int) -> str:
        pieces: List[str] = []
        total = 0
        for d in docs:
            chunk = d.page_content or ""
            if not chunk:
                continue
            # lightweight source header
            src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
            header = f"[source: {os.path.basename(str(src))}]\n" if src else ""
            seg = header + chunk.strip()
            if total + len(seg) > limit:
                remaining = max(0, limit - total)
                if remaining > 0:
                    seg = seg[:remaining]
                else:
                    break
            pieces.append(seg)
            total += len(seg)
            if total >= limit:
                break
        return "\n\n".join(pieces)

    def execute(self, data: ModuleData) -> ModuleData:
        context = self._build_context(data.documents, self.max_context_chars)
        res = self.chain({"context": context, "question": data.query})
        data.answer = res["text"].strip()
        return data


# endregion


# region Builder & handler

def build_rag_pipeline(cfg: PipelineConfig, retriever, *, enable_query_rewrite: bool = True,
                       enable_translation: bool = False, ) -> Pipeline:
    llm_zero = _get_llm(temperature=0.0)
    llm_main = _get_llm(temperature=cfg.temperature)

    pipe = Pipeline()
    if enable_query_rewrite:
        pipe.add(QueryRewriteModule(llm_zero))
    if enable_translation:
        pipe.add(TranslatorModule(llm_zero, cfg.target_language))
    pipe.add(RetrieverModule(retriever))
    pipe.add(AnswerModule(llm_main, max_context_chars=cfg.max_context_chars))
    return pipe


def make_handler(rag: Pipeline):
    def handle(query: str) -> Dict[str, Any]:
        data = rag.execute(query=query)
        return {"answer": data.answer, "sources": data.sources, "query_rewritten": data.query_refined}
    return handle

# endregion
