"""semantic-linker: Semantic record linkage and deduplication using embeddings.

A lightweight Python wrapper around ChromaDB + Gemini embeddings for semantic
record linkage and deduplication, with smart caching to avoid re-embedding.

Example:
    >>> from semantic_linker import SimilarityIndex, match, dedupe
    
    >>> # Create a persistent index
    >>> index = SimilarityIndex.create("census", persist_path="./data")
    >>> index.add(
    ...     records=[{"id": "1", "name": "John Smith", "occupation": "Baker"}],
    ...     id_field="id",
    ...     embed_template="{name} - {occupation}",
    ... )
    
    >>> # Quick matching
    >>> matches = match(
    ...     queries=["Johann Schmidt"],
    ...     targets=["John Smith", "Jane Doe"],
    ... )
    
    >>> # Quick deduplication
    >>> dupes = dedupe(["John Smith", "Jon Smith", "Jane Doe"], threshold=0.8)
"""

from .index import SimilarityIndex, dedupe, match

__version__ = "0.1.0"
__all__ = ["SimilarityIndex", "match", "dedupe"]
