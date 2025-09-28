import os

from pydantic import SecretStr

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