"""Tests for SimilarityIndex class."""

import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from semantic_linker import SimilarityIndex


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
def temp_dir():
    """Fixture to create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def unique_name(prefix: str = "test") -> str:
    """Generate a unique collection name for testing."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class TestCreateIndex:
    """Tests for index creation."""
    
    def test_create_persistent_index(self, temp_dir, mock_embeddings):
        """Test creating a persistent index."""
        index = SimilarityIndex.create("test_index", persist_path=temp_dir)
        
        assert index.name == "test_index"
        assert index.persist_path == temp_dir
        assert index.count == 0
        assert index.embedded_ids == []
    
    def test_create_ephemeral_index(self, mock_embeddings):
        """Test creating an ephemeral (in-memory) index."""
        index = SimilarityIndex.create("ephemeral_index", persist_path=None)
        
        assert index.name == "ephemeral_index"
        assert index.persist_path is None
        assert index.count == 0
    
    def test_load_existing_index(self, temp_dir, mock_embeddings):
        """Test loading an existing index."""
        # Create and populate an index
        index = SimilarityIndex.create("test_index", persist_path=temp_dir)
        index.add(
            records=[{"id": "1", "text": "Hello world"}],
            id_field="id",
            embed_template="{text}",
        )
        
        # Load the index
        loaded = SimilarityIndex.load(persist_path=temp_dir, name="test_index")
        
        assert loaded.name == "test_index"
        assert loaded.count == 1
        assert "1" in loaded.embedded_ids
    
    def test_load_infers_single_collection(self, temp_dir, mock_embeddings):
        """Test loading without name when single collection exists."""
        # Create an index
        index = SimilarityIndex.create("only_collection", persist_path=temp_dir)
        index.add(
            records=[{"id": "1", "text": "Test"}],
            id_field="id",
            embed_template="{text}",
        )
        
        # Load without specifying name
        loaded = SimilarityIndex.load(persist_path=temp_dir)
        
        assert loaded.name == "only_collection"
    
    def test_load_missing_path_raises(self):
        """Test loading from non-existent path raises error."""
        with pytest.raises(ValueError, match="Path does not exist"):
            SimilarityIndex.load("/nonexistent/path")


class TestAddRecords:
    """Tests for adding records."""
    
    def test_add_records(self, mock_embeddings):
        """Test adding records to an index."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        records = [
            {"id": "1", "name": "John Smith"},
            {"id": "2", "name": "Jane Doe"},
        ]
        
        count = index.add(
            records=records,
            id_field="id",
            embed_template="Name: {name}",
        )
        
        assert count == 2
        assert index.count == 2
        assert set(index.embedded_ids) == {"1", "2"}
    
    def test_add_skips_duplicates(self, mock_embeddings):
        """Test that adding duplicate texts is skipped (smart caching)."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        # Add initial record
        count1 = index.add(
            records=[{"id": "1", "name": "John Smith"}],
            id_field="id",
            embed_template="Name: {name}",
        )
        assert count1 == 1
        
        # Try to add same text with different ID
        count2 = index.add(
            records=[{"id": "2", "name": "John Smith"}],
            id_field="id",
            embed_template="Name: {name}",
        )
        assert count2 == 0  # Should skip because text is identical
        assert index.count == 1  # Only original record
    
    def test_add_with_metadata(self, mock_embeddings):
        """Test adding records with metadata fields."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        records = [
            {"id": "1", "name": "John Smith", "age": 30, "city": "NYC"},
        ]
        
        index.add(
            records=records,
            id_field="id",
            embed_template="Name: {name}",
            metadata_fields=["age", "city"],
        )
        
        # Query to check metadata is stored
        results = index.query(
            records=[{"name": "John"}],
            embed_template="Name: {name}",
            top_k=1,
        )
        
        assert len(results) == 1
        assert results.iloc[0]["age"] == 30
        assert results.iloc[0]["city"] == "NYC"
    
    def test_add_empty_list(self, mock_embeddings):
        """Test adding empty list returns 0."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        count = index.add(records=[], id_field="id", embed_template="{text}")
        assert count == 0


class TestCountAndIds:
    """Tests for count and embedded_ids properties."""
    
    def test_count_and_ids(self, mock_embeddings):
        """Test count and embedded_ids properties."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        assert index.count == 0
        assert index.embedded_ids == []
        
        index.add(
            records=[
                {"id": "a", "text": "First"},
                {"id": "b", "text": "Second"},
            ],
            id_field="id",
            embed_template="{text}",
        )
        
        assert index.count == 2
        assert set(index.embedded_ids) == {"a", "b"}


class TestClearAndDelete:
    """Tests for clear and delete methods."""
    
    def test_clear(self, mock_embeddings):
        """Test clearing all records."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        index.add(
            records=[
                {"id": "1", "text": "First"},
                {"id": "2", "text": "Second"},
            ],
            id_field="id",
            embed_template="{text}",
        )
        
        assert index.count == 2
        
        index.clear()
        
        assert index.count == 0
        assert index.embedded_ids == []
    
    def test_delete_by_ids(self, mock_embeddings):
        """Test deleting specific records by ID."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        index.add(
            records=[
                {"id": "1", "text": "First"},
                {"id": "2", "text": "Second"},
                {"id": "3", "text": "Third"},
            ],
            id_field="id",
            embed_template="{text}",
        )
        
        deleted = index.delete(["1", "3"])
        
        assert deleted == 2
        assert index.count == 1
        assert index.embedded_ids == ["2"]
    
    def test_delete_updates_hash_cache(self, mock_embeddings):
        """Test that delete updates hash cache so text can be re-added."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        
        # Add a record
        index.add(
            records=[{"id": "1", "text": "Test"}],
            id_field="id",
            embed_template="{text}",
        )
        
        # Delete it
        index.delete(["1"])
        
        # Should be able to add same text again
        count = index.add(
            records=[{"id": "2", "text": "Test"}],
            id_field="id",
            embed_template="{text}",
        )
        
        assert count == 1
    
    def test_delete_empty_list(self, mock_embeddings):
        """Test deleting empty list returns 0."""
        index = SimilarityIndex.create(unique_name(), persist_path=None)
        deleted = index.delete([])
        assert deleted == 0
