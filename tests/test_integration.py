"""Integration (smoke) tests using the real Gemini API.

Goal: verify we can successfully call the Gemini embeddings API end-to-end.

These tests require a valid API key via:
- GEMINI_API_KEY (literal key)
Run with: uv run pytest tests/test_integration.py -v
"""

import os
import tempfile
import uuid
from pathlib import Path

import polars as pl
import pytest

from semantic_linker import SimilarityIndex, dedupe, match
from semantic_linker.embeddings import embed_texts


def _api_key_available() -> bool:
    key = os.environ.get("GEMINI_API_KEY")
    return bool(key and key.strip())


# Mark as integration and skip if no API key is set.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _api_key_available(),
        reason="No API key found (set GEMINI_API_KEY)",
    ),
]


def unique_name(prefix: str = "test") -> str:
    """Generate a unique collection name for testing."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class TestEmbeddingsIntegration:
    """Smoke tests for real embedding calls."""

    def test_embed_and_query_similar_texts(self):
        """Smoke test: can embed docs and run a query."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)

        # Add some target records
        index.add(
            records=[
                {"id": "1", "text": "The quick brown fox jumps over the lazy dog"},
                {"id": "2", "text": "A fast auburn fox leaps above a sleepy canine"},
                {"id": "3", "text": "Python is a programming language"},
                {"id": "4", "text": "Machine learning uses neural networks"},
            ],
            id_field="id",
            embed_template="{text}",
        )

        # Query with similar text
        results = index.query(
            records=[{"text": "A swift fox jumping over a tired dog"}],
            embed_template="{text}",
            top_k=4,
        )

        # Basic sanity checks: right shape and score range.
        assert isinstance(results, pl.DataFrame)
        assert len(results) == 4
        assert set(["query_id", "query_text", "target_id", "target_text", "similarity", "rank"]).issubset(
            results.columns
        )
        assert results["rank"].min() >= 1
        assert results["rank"].max() <= 4
        assert ((results["similarity"] >= 0.0) & (results["similarity"] <= 1.0)).all()

    def test_embed_batch_smoke(self):
        """Smoke test: batch embeddings path works (gemini-batch)."""
        texts = [
            "Hello world",
            "Goodbye world",
            "A short embedding test",
        ]

        embeddings = embed_texts(
            texts,
            task_type="SEMANTIC_SIMILARITY",
            force_batch=True,
        )

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert all(isinstance(vec, list) and len(vec) > 10 for vec in embeddings)
        assert all(all(isinstance(v, (int, float)) for v in vec) for vec in embeddings)


class TestMatchIntegration:
    """Smoke tests for convenience helpers."""

    def test_match_names(self):
        """Smoke test: match() runs and returns expected schema."""
        results = match(
            queries=["Johann Sebastian Bach", "Wolfgang Mozart", "Ludwig Beethoven"],
            targets=[
                "Johann S. Bach",
                "W.A. Mozart",
                "L. van Beethoven",
                "Franz Schubert",
                "Johannes Brahms",
            ],
            top_k=2,
        )

        assert isinstance(results, pl.DataFrame)
        assert len(results) == 6  # 3 queries × 2 top_k
        assert set(["query_id", "target_id", "similarity", "rank"]).issubset(results.columns)
        assert ((results["similarity"] >= 0.0) & (results["similarity"] <= 1.0)).all()

    def test_match_with_threshold(self):
        """Smoke test: threshold filtering executes (non-strict assertions)."""
        results = match(
            queries=["cat"],
            targets=["kitten", "dog", "automobile", "feline"],
            top_k=4,
            threshold=0.5,
        )

        # Should find cat-related terms above threshold
        assert isinstance(results, pl.DataFrame)
        # All returned results should be >= threshold
        if len(results) > 0:
            assert (results["similarity"] >= 0.5).all()


class TestDedupeIntegration:
    """Smoke tests for dedupe() with real embeddings."""

    def test_dedupe_similar_texts(self):
        """Smoke test: dedupe() runs and returns expected schema."""
        dupes = dedupe(
            texts=[
                "The United States of America",
                "USA",
                "United States",
                "Germany",
                "Federal Republic of Germany",
                "Japan",
            ],
            threshold=0.7,
        )

        assert isinstance(dupes, pl.DataFrame)
        # May be empty depending on model behavior + threshold; just verify schema.
        if len(dupes) > 0:
            assert set(["id_1", "text_1", "id_2", "text_2", "similarity"]).issubset(dupes.columns)
            assert ((dupes["similarity"] >= 0.0) & (dupes["similarity"] <= 1.0)).all()

    def test_dedupe_records_with_template(self):
        """Smoke test: dedupe() works for record dicts with embed_template."""
        dupes = dedupe(
            texts=[
                {"id": "occ1", "german": "Bäcker", "english": "Baker"},
                {"id": "occ2", "german": "Becker", "english": "Baker"},
                {"id": "occ3", "german": "Lehrer", "english": "Teacher"},
            ],
            embed_template="German: {german}, English: {english}",
            threshold=0.8,
        )
        assert isinstance(dupes, pl.DataFrame)
        if len(dupes) > 0:
            assert set(["id_1", "id_2", "similarity"]).issubset(dupes.columns)
            assert ((dupes["similarity"] >= 0.0) & (dupes["similarity"] <= 1.0)).all()


class TestPersistentIndexIntegration:
    """Smoke test: persistence works with real embeddings."""

    def test_create_load_persist(self):
        """Smoke test: can create, persist, load, and query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir)

            # Create and populate index
            index = SimilarityIndex.create("census", persist_path=persist_path)
            index.add(
                records=[
                    {"id": "p1", "name": "John Smith", "occupation": "Blacksmith"},
                    {"id": "p2", "name": "Mary Johnson", "occupation": "Teacher"},
                ],
                id_field="id",
                embed_template="{name}, {occupation}",
                metadata_fields=["occupation"],
            )

            assert index.count == 2

            # Load the index
            loaded = SimilarityIndex.load(persist_path, name="census")
            assert loaded.count == 2
            assert set(loaded.embedded_ids) == {"p1", "p2"}

            # Query the loaded index
            results = loaded.query(
                records=[{"name": "Jon Smith", "occupation": "Smith"}],
                embed_template="{name}, {occupation}",
                top_k=2,
            )

            assert len(results) == 2
            assert set(["query_id", "target_id", "similarity", "rank"]).issubset(results.columns)

    def test_smart_caching(self):
        """Smoke test: caching prevents re-adding identical texts."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)

        # Add initial records
        count1 = index.add(
            records=[
                {"id": "1", "text": "Hello world"},
                {"id": "2", "text": "Goodbye world"},
            ],
            id_field="id",
            embed_template="{text}",
        )
        assert count1 == 2

        # Try to add same texts with different IDs - should be skipped
        count2 = index.add(
            records=[
                {"id": "3", "text": "Hello world"},  # Same as id=1
                {"id": "4", "text": "New text"},  # Actually new
            ],
            id_field="id",
            embed_template="{text}",
        )
        assert count2 == 1  # Only the new text should be added
        assert index.count == 3
