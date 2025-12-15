"""Tests for query and match functionality."""

import uuid
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from semantic_linker import SimilarityIndex, match


def unique_name(prefix: str = "test") -> str:
    """Generate a unique collection name for testing."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# Mock embedding function for testing
def mock_embed_texts(texts, task_type="SEMANTIC_SIMILARITY", force_batch=False):
    """Return deterministic embeddings based on text hash."""
    embeddings = []
    for text in texts:
        # Create a simple deterministic embedding based on text
        np.random.seed(hash(text) % (2**32))
        embeddings.append(np.random.randn(768).tolist())
    return embeddings


@pytest.fixture
def mock_embeddings():
    """Fixture to mock the embed_texts function."""
    with patch("semantic_linker.index.embed_texts", side_effect=mock_embed_texts):
        yield


@pytest.fixture
def populated_index(mock_embeddings):
    """Fixture to create a populated index for testing."""
    index = SimilarityIndex.create(unique_name(), persist_path=None)
    
    records = [
        {"id": "t1", "name": "John Smith", "occupation": "Baker"},
        {"id": "t2", "name": "Jane Doe", "occupation": "Teacher"},
        {"id": "t3", "name": "Bob Johnson", "occupation": "Doctor"},
    ]
    
    index.add(
        records=records,
        id_field="id",
        embed_template="Name: {name}, Job: {occupation}",
        metadata_fields=["occupation"],
    )
    
    return index


class TestQueryBasic:
    """Tests for basic query functionality."""
    
    def test_query_basic(self, populated_index, mock_embeddings):
        """Test basic query returns expected structure."""
        results = populated_index.query(
            records=[{"id": "q1", "name": "John", "occupation": "Baker"}],
            embed_template="Name: {name}, Job: {occupation}",
            top_k=3,
        )
        
        assert isinstance(results, pl.DataFrame)
        assert len(results) == 3
        assert "query_id" in results.columns
        assert "query_text" in results.columns
        assert "target_id" in results.columns
        assert "target_text" in results.columns
        assert "similarity" in results.columns
        assert "rank" in results.columns
        
        # Check ranks are 1, 2, 3
        assert list(results["rank"]) == [1, 2, 3]
        
        # Check query_id is correct
        assert all(results["query_id"] == "q1")
    
    def test_query_auto_generates_id(self, populated_index, mock_embeddings):
        """Test query auto-generates ID if not provided."""
        results = populated_index.query(
            records=[{"name": "John", "occupation": "Baker"}],
            embed_template="Name: {name}, Job: {occupation}",
            top_k=1,
        )
        
        assert results.row(0, named=True)["query_id"] == "q_0"
    
    def test_query_empty_index(self, mock_embeddings):
        """Test querying empty index returns empty DataFrame."""
        index = SimilarityIndex.create(unique_name("empty"), persist_path=None)
        
        results = index.query(
            records=[{"text": "test"}],
            embed_template="{text}",
        )

        assert isinstance(results, pl.DataFrame)
        assert len(results) == 0
        assert "query_id" in results.columns


class TestQueryWithThreshold:
    """Tests for query with threshold filtering."""
    
    def test_query_with_threshold(self, populated_index, mock_embeddings):
        """Test query with high threshold returns fewer results."""
        # With a very high threshold, we should get fewer results
        results = populated_index.query(
            records=[{"name": "Unknown", "occupation": "Unknown"}],
            embed_template="Name: {name}, Job: {occupation}",
            top_k=3,
            threshold=0.99,  # Very high threshold
        )
        
        # All results should have similarity >= threshold
        if len(results) > 0:
            assert (results["similarity"] >= 0.99).all()
    
    def test_query_threshold_none_returns_all(self, populated_index, mock_embeddings):
        """Test query with no threshold returns top_k results."""
        results = populated_index.query(
            records=[{"name": "Test", "occupation": "Test"}],
            embed_template="Name: {name}, Job: {occupation}",
            top_k=3,
            threshold=None,
        )
        
        assert len(results) == 3


class TestQueryWithMetadata:
    """Tests for query returning metadata."""
    
    def test_query_returns_metadata(self, populated_index, mock_embeddings):
        """Test that stored metadata is returned in query results."""
        results = populated_index.query(
            records=[{"name": "John", "occupation": "Baker"}],
            embed_template="Name: {name}, Job: {occupation}",
            top_k=3,
        )
        
        # Should have occupation metadata column
        assert "occupation" in results.columns
        
        # Check metadata values are correct
        target_occupations = set(results["occupation"])
        assert target_occupations == {"Baker", "Teacher", "Doctor"}


class TestMatchConvenienceFunction:
    """Tests for the match convenience function."""
    
    def test_match_with_strings(self, mock_embeddings):
        """Test match with simple string lists."""
        results = match(
            queries=["John Smith", "Jane Doe"],
            targets=["Johann Schmidt", "Jan Doe", "Bob"],
            top_k=2,
        )

        assert isinstance(results, pl.DataFrame)
        assert len(results) == 4  # 2 queries × 2 top_k
        
        # Check auto-generated IDs
        assert set(results["query_id"]) == {"q_0", "q_1"}
        assert set(results["target_id"]).issubset({"t_0", "t_1", "t_2"})
    
    def test_match_with_dicts(self, mock_embeddings):
        """Test match with dict records."""
        results = match(
            queries=[
                {"id": "q1", "name": "John", "city": "NYC"},
            ],
            targets=[
                {"id": "t1", "name": "Johann", "city": "Berlin"},
                {"id": "t2", "name": "Jane", "city": "Paris"},
            ],
            embed_template="Name: {name}, City: {city}",
            top_k=2,
        )
        
        assert len(results) == 2
        assert results.row(0, named=True)["query_id"] == "q1"
        assert set(results["target_id"]) == {"t1", "t2"}
    
    def test_match_requires_template_for_dicts(self, mock_embeddings):
        """Test that match raises error if template not provided for dicts."""
        with pytest.raises(ValueError, match="embed_template is required"):
            match(
                queries=[{"name": "John"}],
                targets=[{"name": "Jane"}],
            )
    
    def test_match_with_threshold(self, mock_embeddings):
        """Test match with threshold filtering."""
        results = match(
            queries=["Test"],
            targets=["Target1", "Target2"],
            top_k=2,
            threshold=0.99,  # Very high threshold
        )
        
        # All results should meet threshold
        if len(results) > 0:
            assert (results["similarity"] >= 0.99).all()


class TestMultipleQueries:
    """Tests for querying with multiple records."""
    
    def test_multiple_queries(self, populated_index, mock_embeddings):
        """Test querying with multiple records at once."""
        results = populated_index.query(
            records=[
                {"id": "q1", "name": "John", "occupation": "Baker"},
                {"id": "q2", "name": "Jane", "occupation": "Teacher"},
            ],
            embed_template="Name: {name}, Job: {occupation}",
            top_k=2,
        )
        
        assert len(results) == 4  # 2 queries × 2 top_k
        assert set(results["query_id"]) == {"q1", "q2"}

        # Each query should have ranks 1 and 2
        q1_results = results.filter(pl.col("query_id") == "q1")
        q2_results = results.filter(pl.col("query_id") == "q2")

        assert list(q1_results["rank"]) == [1, 2]
        assert list(q2_results["rank"]) == [1, 2]
