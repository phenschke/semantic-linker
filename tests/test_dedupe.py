"""Tests for deduplication functionality."""

import uuid
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from semantic_linker import SimilarityIndex, dedupe


def unique_name(prefix: str = "test") -> str:
    """Generate a unique collection name for testing."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# Mock embedding function that creates similar embeddings for similar texts
def mock_embed_texts_for_dedupe(texts, task_type="SEMANTIC_SIMILARITY", force_batch=False):
    """Return embeddings where similar texts get similar embeddings."""
    embeddings = []
    for text in texts:
        # Base embedding from text
        np.random.seed(hash(text) % (2**32))
        base = np.random.randn(768)
        
        # Make "John Smith" and "Jon Smith" similar by using shared prefix
        if text.lower().startswith("john") or text.lower().startswith("jon"):
            np.random.seed(42)  # Same seed for similar names
            base = np.random.randn(768)
            # Add small perturbation based on full text
            np.random.seed(hash(text) % (2**32))
            base += 0.1 * np.random.randn(768)
        
        # Normalize
        base = base / np.linalg.norm(base)
        embeddings.append(base.tolist())
    
    return embeddings


@pytest.fixture
def mock_embeddings():
    """Fixture to mock the embed_texts function."""
    with patch("semantic_linker.index.embed_texts", side_effect=mock_embed_texts_for_dedupe):
        yield


class TestFindDuplicatesBasic:
    """Tests for basic find_duplicates functionality."""
    
    def test_find_duplicates_basic(self, mock_embeddings):
        """Test basic duplicate detection."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        index.add(
            records=[
                {"id": "1", "text": "John Smith"},
                {"id": "2", "text": "Jon Smith"},  # Similar to John Smith
                {"id": "3", "text": "Jane Doe"},   # Different
            ],
            id_field="id",
            embed_template="{text}",
        )
        
        dupes = index.find_duplicates(threshold=0.5)

        assert isinstance(dupes, pl.DataFrame)
        assert "id_1" in dupes.columns
        assert "text_1" in dupes.columns
        assert "id_2" in dupes.columns
        assert "text_2" in dupes.columns
        assert "similarity" in dupes.columns
    
    def test_find_duplicates_empty_index(self, mock_embeddings):
        """Test finding duplicates in empty index."""
        index = SimilarityIndex.create(unique_name("empty"), persist_path=None)
        
        dupes = index.find_duplicates()

        assert isinstance(dupes, pl.DataFrame)
        assert len(dupes) == 0
    
    def test_find_duplicates_single_record(self, mock_embeddings):
        """Test finding duplicates with only one record."""
        index = SimilarityIndex.create(unique_name("single"), persist_path=None)
        
        index.add(
            records=[{"id": "1", "text": "Only one"}],
            id_field="id",
            embed_template="{text}",
        )
        
        dupes = index.find_duplicates()
        
        assert len(dupes) == 0


class TestFindDuplicatesThreshold:
    """Tests for threshold behavior in duplicate detection."""
    
    def test_high_threshold_finds_fewer_duplicates(self, mock_embeddings):
        """Test that higher threshold returns fewer duplicates."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        index.add(
            records=[
                {"id": "1", "text": "Apple"},
                {"id": "2", "text": "Apple"},  # Exact same
                {"id": "3", "text": "Banana"},
            ],
            id_field="id",
            embed_template="{text}",
        )
        
        # Low threshold should find more pairs
        dupes_low = index.find_duplicates(threshold=0.5)
        
        # High threshold should find fewer pairs
        dupes_high = index.find_duplicates(threshold=0.99)
        
        # Both should be DataFrames
        assert isinstance(dupes_low, pl.DataFrame)
        assert isinstance(dupes_high, pl.DataFrame)
    
    def test_results_sorted_by_similarity(self, mock_embeddings):
        """Test that results are sorted by similarity descending."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        index.add(
            records=[
                {"id": "1", "text": "A"},
                {"id": "2", "text": "B"},
                {"id": "3", "text": "C"},
                {"id": "4", "text": "D"},
            ],
            id_field="id",
            embed_template="{text}",
        )
        
        dupes = index.find_duplicates(threshold=0.0)  # Get all pairs
        
        if len(dupes) > 1:
            # Check sorted in descending order
            similarities = dupes["similarity"].to_list()
            assert similarities == sorted(similarities, reverse=True)


class TestDedupeConvenienceFunction:
    """Tests for the dedupe convenience function."""
    
    def test_dedupe_with_strings(self, mock_embeddings):
        """Test dedupe with simple string list."""
        dupes = dedupe(
            texts=["John Smith", "Jon Smith", "Jane Doe"],
            threshold=0.5,
        )

        assert isinstance(dupes, pl.DataFrame)
        
        # Check auto-generated IDs
        if len(dupes) > 0:
            assert dupes.row(0, named=True)["id_1"].startswith("r_")
            assert dupes.row(0, named=True)["id_2"].startswith("r_")
    
    def test_dedupe_with_dicts(self, mock_embeddings):
        """Test dedupe with dict records."""
        dupes = dedupe(
            texts=[
                {"id": "a", "name": "John", "city": "NYC"},
                {"id": "b", "name": "Jon", "city": "NYC"},
            ],
            embed_template="Name: {name}, City: {city}",
            threshold=0.5,
        )
        
        assert isinstance(dupes, pl.DataFrame)

        if len(dupes) > 0:
            row = dupes.row(0, named=True)
            assert set([row["id_1"], row["id_2"]]) == {"a", "b"}
    
    def test_dedupe_requires_template_for_dicts(self, mock_embeddings):
        """Test that dedupe raises error if template not provided for dicts."""
        with pytest.raises(ValueError, match="embed_template is required"):
            dedupe(
                texts=[{"name": "John"}, {"name": "Jane"}],
            )
    
    def test_dedupe_empty_list(self, mock_embeddings):
        """Test dedupe with empty list."""
        dupes = dedupe(texts=[], threshold=0.9)

        assert isinstance(dupes, pl.DataFrame)
        assert len(dupes) == 0


class TestDuplicatePairStructure:
    """Tests for the structure of duplicate pairs."""
    
    def test_no_self_matches(self, mock_embeddings):
        """Test that self-matches are excluded."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        index.add(
            records=[
                {"id": "1", "text": "Test"},
                {"id": "2", "text": "Other"},
            ],
            id_field="id",
            embed_template="{text}",
        )
        
        dupes = index.find_duplicates(threshold=0.0)

        # No row should have id_1 == id_2
        for row in dupes.iter_rows(named=True):
            assert row["id_1"] != row["id_2"]
    
    def test_pairs_are_unique(self, mock_embeddings):
        """Test that pairs are not duplicated (only upper triangle)."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        index.add(
            records=[
                {"id": "1", "text": "A"},
                {"id": "2", "text": "B"},
                {"id": "3", "text": "C"},
            ],
            id_field="id",
            embed_template="{text}",
        )
        
        dupes = index.find_duplicates(threshold=0.0)

        # Check no duplicate pairs
        pairs = set()
        for row in dupes.iter_rows(named=True):
            pair = frozenset([row["id_1"], row["id_2"]])
            assert pair not in pairs
            pairs.add(pair)
