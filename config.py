import os
from pathlib import Path

from pydantic import SecretStr

DOCS_DIR = os.getenv("DOCS_DIR", os.path.join(Path(__file__).parent, "./data/docs"))
FAISS_DIR = os.getenv("FAISS_DIR", os.path.join(Path(__file__).parent, "./data/faiss_index"))

# LLM provider selection:
#   LLM_PROVIDER=ollama (default) or openai
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI/vLLM-compatible settings
OPENAI_API_KEY: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", "not-needed-for-local"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Groq settings
GROQ_API_KEY: SecretStr = SecretStr(
    os.getenv("GROQ_API_KEY", Path(Path(__file__).parent / ".groq_api_key").read_text(encoding="utf-8-sig"))
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Embeddings model
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")