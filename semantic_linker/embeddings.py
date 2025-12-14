"""Embedding helpers for semantic-linker using Gemini."""

from __future__ import annotations

import os
from typing import Literal

from google import genai
from google.genai import types

# Threshold for switching to batch API
BATCH_THRESHOLD = 100


TaskType = Literal[
    "SEMANTIC_SIMILARITY",
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "CLASSIFICATION",
    "CLUSTERING",
]


def embed_texts(
    texts: list[str],
    task_type: TaskType = "SEMANTIC_SIMILARITY",
    force_batch: bool = False,
) -> list[list[float]]:
    """Embed texts using Gemini.
    
    Auto-selects batch vs real-time based on count.
    Uses batch API for 50% cost savings when processing >= 100 texts.
    
    Args:
        texts: List of texts to embed.
        task_type: The embedding task type. One of:
            - SEMANTIC_SIMILARITY: For comparing text similarity
            - RETRIEVAL_QUERY: For query embeddings in search
            - RETRIEVAL_DOCUMENT: For document embeddings in search
            - CLASSIFICATION: For text classification
            - CLUSTERING: For clustering texts
        force_batch: Force batch API even for small sets.
        
    Returns:
        List of embedding vectors (list of floats).
        
    Raises:
        ValueError: If texts is empty.
        RuntimeError: If API call fails.
    """
    if not texts:
        return []
    
    use_batch = force_batch or len(texts) >= BATCH_THRESHOLD
    
    if use_batch:
        return _embed_batch(texts, task_type)
    else:
        return _embed_realtime(texts, task_type)


def _resolve_api_key() -> str | None:
    """Resolve the Gemini API key.

    Supported env vars:
    - GEMINI_API_KEY: literal API key
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    return key.strip() or None


def _embed_realtime(texts: list[str], task_type: TaskType) -> list[list[float]]:
    """Embed texts using real-time API (for small batches).
    
    Args:
        texts: List of texts to embed.
        task_type: The embedding task type.
        
    Returns:
        List of embedding vectors.
    """
    api_key = _resolve_api_key()
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    embeddings = []
    
    for text in texts:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embeddings.append(list(response.embeddings[0].values))
    
    return embeddings


def _embed_batch(texts: list[str], task_type: TaskType) -> list[list[float]]:
    """Embed texts using batch API (for large batches, 50% cost savings).
    
    Args:
        texts: List of texts to embed.
        task_type: The embedding task type.
        
    Returns:
        List of embedding vectors.
    """
    from gemini_batch import batch_embed
    
    return batch_embed(texts=texts, task_type=task_type)
