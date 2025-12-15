"""SimilarityIndex class for semantic record linkage and deduplication."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import embed_texts
from .utils import compute_text_hash, format_record


def _unique_name(prefix: str = "temp") -> str:
    """Generate a unique collection name for ephemeral indexes."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class SimilarityIndex:
    """A persistent or ephemeral index for semantic similarity search.
    
    The SimilarityIndex stores embeddings in ChromaDB and supports:
    - Smart caching: avoids re-embedding duplicate texts
    - Multi-field embedding via format templates
    - Efficient similarity queries and deduplication
    
    Example:
        >>> index = SimilarityIndex.create("my_index", persist_path="./data")
        >>> index.add(
        ...     records=[{"id": "1", "name": "John Smith"}],
        ...     id_field="id",
        ...     embed_template="{name}",
        ... )
        1
        >>> results = index.query(
        ...     records=[{"name": "Jon Smith"}],
        ...     embed_template="{name}",
        ... )
    """
    
    def __init__(
        self,
        collection: chromadb.Collection,
        persist_path: Path | None,
        name: str,
    ) -> None:
        """Initialize SimilarityIndex (use create() or load() instead).
        
        Args:
            collection: ChromaDB collection instance.
            persist_path: Path to ChromaDB persistence directory.
            name: Name of the collection.
        """
        self._collection = collection
        self._persist_path = persist_path
        self._name = name
        self._hash_cache: set[str] = set()
        self._load_hash_cache()
    
    def _load_hash_cache(self) -> None:
        """Load existing text hashes from ChromaDB metadata."""
        if self._collection.count() > 0:
            all_data = self._collection.get(include=["metadatas"])
            self._hash_cache = { # type: ignore
                m["_text_hash"] 
                for m in all_data["metadatas"]  # type: ignore
                if m and "_text_hash" in m
            }
    
    @classmethod
    def create(
        cls,
        name: str,
        persist_path: str | Path | None = None,
    ) -> SimilarityIndex:
        """Create a new index.
        
        Args:
            name: Collection name (used as identifier).
            persist_path: Directory to store ChromaDB data. If None, creates
                an ephemeral in-memory index.
                
        Returns:
            A new SimilarityIndex instance.
            
        Example:
            >>> # Persistent index
            >>> index = SimilarityIndex.create("census", persist_path="./data")
            >>> # Ephemeral index
            >>> index = SimilarityIndex.create("temp", persist_path=None)
        """
        if persist_path is not None:
            persist_path = Path(persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_path))
        else:
            client = chromadb.EphemeralClient()
        
        collection = client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:batch_size": 100000,
                "hnsw:sync_threshold": 100000,
            }
        )
        
        return cls(collection, persist_path, name)
    
    @classmethod
    def load(
        cls,
        persist_path: str | Path,
        name: str | None = None,
    ) -> SimilarityIndex:
        """Load an existing index from disk.
        
        Args:
            persist_path: Directory containing ChromaDB data.
            name: Collection name. If None and only one collection exists,
                it will be loaded automatically.
                
        Returns:
            The loaded SimilarityIndex instance.
            
        Raises:
            ValueError: If name is None and multiple collections exist.
            ValueError: If the specified collection doesn't exist.
        """
        persist_path = Path(persist_path)
        if not persist_path.exists():
            raise ValueError(f"Path does not exist: {persist_path}")
        
        client = chromadb.PersistentClient(path=str(persist_path))
        
        if name is None:
            collections = client.list_collections()
            if len(collections) == 0:
                raise ValueError(f"No collections found at {persist_path}")
            elif len(collections) > 1:
                names = [c.name for c in collections]
                raise ValueError(
                    f"Multiple collections found: {names}. Specify 'name' parameter."
                )
            name = collections[0].name
        
        try:
            collection = client.get_collection(name)
        except Exception as e:
            raise ValueError(f"Collection '{name}' not found at {persist_path}") from e
        
        return cls(collection, persist_path, name)
    
    @property
    def count(self) -> int:
        """Number of records in the index."""
        return self._collection.count()
    
    @property
    def embedded_ids(self) -> list[str]:
        """List of all record IDs in the index."""
        if self.count == 0:
            return []
        return self._collection.get()["ids"]
    
    @property
    def name(self) -> str:
        """Collection name."""
        return self._name
    
    @property
    def persist_path(self) -> Path | None:
        """Storage location (None for ephemeral indexes)."""
        return self._persist_path
    
    def add(
        self,
        records: list[dict[str, Any]],
        id_field: str = "id",
        embed_template: str = "{text}",
        metadata_fields: list[str] | None = None,
        force_batch: bool = False,
    ) -> int:
        """Add records to the index.
        
        Smart behavior:
        - Computes SHA256 hash of formatted text
        - Skips records whose text hash already exists
        - Only embeds truly new records
        
        Args:
            records: List of record dictionaries.
            id_field: Field name containing unique record ID.
            embed_template: Format string for embedding text.
                Example: "Name: {name} | Occupation: {job}"
            metadata_fields: Additional fields to store for retrieval.
            force_batch: Force batch API even for small sets.
            
        Returns:
            Number of NEW records added (not total).
            
        Example:
            >>> index.add(
            ...     records=[{"id": "1", "name": "John", "age": 30}],
            ...     id_field="id",
            ...     embed_template="Name: {name}",
            ...     metadata_fields=["age"],
            ... )
            1
        """
        metadata_fields = metadata_fields or []
        
        # Format texts and compute hashes
        new_records = []
        for record in records:
            text = format_record(record, embed_template)
            text_hash = compute_text_hash(text)
            
            if text_hash not in self._hash_cache:
                new_records.append({
                    "id": str(record[id_field]),
                    "text": text,
                    "hash": text_hash,
                    "metadata": {k: record.get(k) for k in metadata_fields},
                })
                self._hash_cache.add(text_hash)
        
        if not new_records:
            return 0
        
        # Embed new texts
        texts = [r["text"] for r in new_records]
        embeddings = embed_texts(
            texts, 
            task_type="RETRIEVAL_DOCUMENT",
            force_batch=force_batch,
        )
        
        # Add to ChromaDB
        self._collection.add(
            ids=[r["id"] for r in new_records],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"_text_hash": r["hash"], "_text": r["text"], **r["metadata"]}
                for r in new_records
            ],
        )
        
        return len(new_records)
    
    def query(
        self,
        records: list[dict[str, Any]],
        embed_template: str = "{text}",
        top_k: int = 5,
        threshold: float | None = None,
        force_batch: bool = False,
    ) -> pl.DataFrame:
        """Query the index for similar records.
        
        Args:
            records: Query records (list of dicts).
            embed_template: Format string (should match add template).
            top_k: Number of matches per query.
            threshold: Minimum similarity score (0-1). If None, no filtering.
            force_batch: Force batch API for query embeddings.
            
        Returns:
            DataFrame with columns:
                - query_id: ID of query record
                - query_text: formatted query text
                - target_id: ID of matched record
                - target_text: formatted target text
                - similarity: cosine similarity score (0-1)
                - rank: 1-indexed rank of match
                - [metadata fields]: any stored metadata
                
        Example:
            >>> results = index.query(
            ...     records=[{"id": "q1", "name": "John Smith"}],
            ...     embed_template="Name: {name}",
            ...     top_k=5,
            ...     threshold=0.7,
            ... )
        """
        if self.count == 0:
            return pl.DataFrame(
                schema={
                    "query_id": pl.String,
                    "query_text": pl.String,
                    "target_id": pl.String,
                    "target_text": pl.String,
                    "similarity": pl.Float64,
                    "rank": pl.Int64,
                }
            )
        
        # Format query texts
        queries = []
        for i, record in enumerate(records):
            text = format_record(record, embed_template)
            record_id = record.get("id", f"q_{i}")
            queries.append({
                "id": str(record_id),
                "text": text,
            })
        
        # Embed queries
        query_texts = [q["text"] for q in queries]
        query_embeddings = embed_texts(
            query_texts,
            task_type="RETRIEVAL_QUERY",
            force_batch=force_batch,
        )
        
        # Get all stored embeddings
        all_data = self._collection.get(include=["embeddings", "metadatas"])
        stored_embeddings = np.array(all_data["embeddings"])
        stored_metadatas = all_data["metadatas"]
        stored_ids = all_data["ids"]
        
        # Compute similarities
        similarities = cosine_similarity(query_embeddings, stored_embeddings)
        
        # Build results
        results = []
        for q_idx, query in enumerate(queries):
            scores = similarities[q_idx]
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            for rank, t_idx in enumerate(top_indices):
                sim = float(scores[t_idx])
                if threshold is not None and sim < threshold:
                    continue
                
                metadata = stored_metadatas[t_idx] or {}
                results.append({
                    "query_id": query["id"],
                    "query_text": query["text"],
                    "target_id": stored_ids[t_idx],
                    "target_text": metadata.get("_text", ""),
                    "similarity": sim,
                    "rank": rank + 1,
                    **{k: v for k, v in metadata.items() if not k.startswith("_")},
                })

        return pl.DataFrame(results)
    
    def find_duplicates(
        self,
        threshold: float = 0.9,
        exclude_self: bool = True,
    ) -> pl.DataFrame:
        """Find duplicate records within the index.
        
        Args:
            threshold: Minimum similarity to consider as duplicate (0-1).
            exclude_self: Exclude exact self-matches (always True for now).
            
        Returns:
            DataFrame with columns:
                - id_1: first record ID
                - text_1: first record text
                - id_2: second record ID
                - text_2: second record text
                - similarity: cosine similarity score
                
        Example:
            >>> dupes = index.find_duplicates(threshold=0.9)
            >>> print(dupes)
        """
        if self.count < 2:
            return pl.DataFrame(
                schema={
                    "id_1": pl.String,
                    "text_1": pl.String,
                    "id_2": pl.String,
                    "text_2": pl.String,
                    "similarity": pl.Float64,
                }
            )
        
        # Get all embeddings
        all_data = self._collection.get(include=["embeddings", "metadatas"])
        embeddings = np.array(all_data["embeddings"])
        metadatas = all_data["metadatas"]
        ids = all_data["ids"]
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings, embeddings)
        
        # Find pairs above threshold (upper triangle only)
        results = []
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(similarities[i, j])
                if sim >= threshold:
                    meta_i = metadatas[i] or {}
                    meta_j = metadatas[j] or {}
                    results.append({
                        "id_1": ids[i],
                        "text_1": meta_i.get("_text", ""),
                        "id_2": ids[j],
                        "text_2": meta_j.get("_text", ""),
                        "similarity": sim,
                    })

        df = pl.DataFrame(results)
        if len(df) > 0:
            df = df.sort("similarity", descending=True)
        return df
    
    def delete(self, ids: list[str]) -> int:
        """Delete records by ID.
        
        Args:
            ids: List of record IDs to delete.
            
        Returns:
            Number of records deleted.
        """
        if not ids:
            return 0
        
        # Get hashes for records being deleted to update cache
        existing = self._collection.get(ids=ids, include=["metadatas"])
        deleted_hashes = {
            m["_text_hash"] 
            for m in existing["metadatas"] 
            if m and "_text_hash" in m
        }
        
        self._collection.delete(ids=ids)
        self._hash_cache -= deleted_hashes
        
        return len(existing["ids"])
    
    def clear(self) -> None:
        """Remove all records from the index."""
        if self.count > 0:
            all_ids = self._collection.get()["ids"]
            self._collection.delete(ids=all_ids)
        self._hash_cache.clear()


def match(
    queries: list[str] | list[dict[str, Any]],
    targets: list[str] | list[dict[str, Any]],
    top_k: int = 5,
    threshold: float | None = None,
    embed_template: str | None = None,
    id_field: str = "id",
) -> pl.DataFrame:
    """Quick matching between two sets of records (ephemeral).
    
    Convenience function for one-off matching without creating a persistent index.
    
    Args:
        queries: Query texts (list of strings) or records (list of dicts).
        targets: Target texts (list of strings) or records (list of dicts).
        top_k: Number of matches per query.
        threshold: Minimum similarity score.
        embed_template: Format string for embedding. Required if using dicts.
        id_field: Field containing unique ID (for dicts).
        
    Returns:
        DataFrame with match results (same format as SimilarityIndex.query).
        
    Example:
        >>> # Simple string matching
        >>> matches = match(
        ...     queries=["Johann Schmidt", "Maria Müller"],
        ...     targets=["John Smith", "Mary Miller", "Hans Schmidt"],
        ...     top_k=3,
        ... )
        
        >>> # Dict matching with template
        >>> matches = match(
        ...     queries=[{"id": "q1", "name": "Johann", "job": "Baker"}],
        ...     targets=[{"id": "t1", "name": "John", "job": "Baker"}],
        ...     embed_template="{name} - {job}",
        ... )
    """
    # Convert strings to dicts if needed
    if queries and isinstance(queries[0], str):
        queries = [{"id": f"q_{i}", "text": t} for i, t in enumerate(queries)]
        if embed_template is None:
            embed_template = "{text}"
    
    if targets and isinstance(targets[0], str):
        targets = [{"id": f"t_{i}", "text": t} for i, t in enumerate(targets)]
        if embed_template is None:
            embed_template = "{text}"
    
    if embed_template is None:
        raise ValueError("embed_template is required when using dict records")
    
    # Create ephemeral index with unique name
    index = SimilarityIndex.create(_unique_name("match"), persist_path=None)
    
    # Add targets
    index.add(
        records=targets,
        id_field=id_field,
        embed_template=embed_template,
    )
    
    # Query with queries
    return index.query(
        records=queries,
        embed_template=embed_template,
        top_k=top_k,
        threshold=threshold,
    )


def dedupe(
    texts: list[str] | list[dict[str, Any]],
    threshold: float = 0.9,
    embed_template: str | None = None,
    id_field: str = "id",
) -> pl.DataFrame:
    """Quick deduplication of a set of records (ephemeral).
    
    Convenience function for one-off deduplication without creating a persistent index.
    
    Args:
        texts: Texts (list of strings) or records (list of dicts).
        threshold: Minimum similarity to consider as duplicate.
        embed_template: Format string for embedding. Required if using dicts.
        id_field: Field containing unique ID (for dicts).
        
    Returns:
        DataFrame with duplicate pairs (same format as SimilarityIndex.find_duplicates).
        
    Example:
        >>> dupes = dedupe(
        ...     texts=["John Smith", "Jon Smith", "Jane Doe"],
        ...     threshold=0.8,
        ... )
    """
    # Handle empty list early
    if not texts:
        return pl.DataFrame(
            schema={
                "id_1": pl.String,
                "text_1": pl.String,
                "id_2": pl.String,
                "text_2": pl.String,
                "similarity": pl.Float64,
            }
        )
    
    # Convert strings to dicts if needed
    if isinstance(texts[0], str):
        texts = [{"id": f"r_{i}", "text": t} for i, t in enumerate(texts)]
        if embed_template is None:
            embed_template = "{text}"
    
    if embed_template is None:
        raise ValueError("embed_template is required when using dict records")
    
    # Create ephemeral index with unique name
    index = SimilarityIndex.create(_unique_name("dedupe"), persist_path=None)
    
    # Add all texts
    index.add(
        records=texts,
        id_field=id_field,
        embed_template=embed_template,
    )
    
    # Find duplicates
    return index.find_duplicates(threshold=threshold)
