# semantic-linker

A lightweight Python wrapper around ChromaDB + Gemini embeddings for semantic record linkage and deduplication, with smart caching to avoid re-embedding.

## Features

- **Semantic Record Linkage**: Match records across datasets using embedding similarity
- **Deduplication**: Find duplicate records within a single dataset
- **Smart Caching**: SHA256-based caching to avoid re-embedding identical texts
- **Multi-Field Embeddings**: Combine multiple fields using format templates
- **Cost-Efficient**: Auto-switches to batch API (50% savings) for large datasets
- **Persistent Storage**: ChromaDB-backed indexes that persist across sessions

## Installation

```bash
uv add semantic-linker
```

Or install from source:

```bash
uv sync
```

## Quick Start

### Simple String Matching

```python
from semantic_linker import match

matches = match(
    queries=["Johann Schmidt", "Maria Müller"],
    targets=["John Smith", "Mary Miller", "Hans Schmidt"],
    top_k=3,
)
print(matches)
```

### Simple Deduplication

```python
from semantic_linker import dedupe

dupes = dedupe(
    texts=["John Smith", "Jon Smith", "Jane Doe"],
    threshold=0.8,
)
print(dupes)
```

## Core API

### SimilarityIndex Class

For persistent, reusable indexes:

```python
from semantic_linker import SimilarityIndex

# Create a persistent index
index = SimilarityIndex.create(
    name="occupations",
    persist_path="./occupation_index",
)

# Add records with multi-field template
index.add(
    records=[
        {"id": "1", "german_name": "Bäcker", "english": "Baker"},
        {"id": "2", "german_name": "Lehrer", "english": "Teacher"},
    ],
    id_field="id",
    embed_template="German: {german_name} | English: {english}",
    metadata_fields=["english"],
)

print(f"Added {index.count} records")

# Query the index
results = index.query(
    records=[{"german_name": "Becker", "english": "Baker"}],
    embed_template="German: {german_name} | English: {english}",
    top_k=5,
    threshold=0.7,
)
print(results)

# Find duplicates within the index
dupes = index.find_duplicates(threshold=0.9)
print(dupes)
```

### Loading an Existing Index

```python
# Load by path and name
index = SimilarityIndex.load(
    persist_path="./occupation_index",
    name="occupations",
)

# Or auto-detect if only one collection exists
index = SimilarityIndex.load(persist_path="./occupation_index")
```

### Ephemeral (In-Memory) Index

```python
index = SimilarityIndex.create(
    name="temp_index",
    persist_path=None,  # In-memory only
)
```

## Use Cases

### Census Record Linkage

```python
from semantic_linker import SimilarityIndex
import pandas as pd

# Load census data
census_1890 = pd.read_csv("census_1890.csv")
census_1900 = pd.read_csv("census_1900.csv")

# Create index for 1890 census
index = SimilarityIndex.create("census_1890", persist_path="./census_data")

index.add(
    records=census_1890.to_dict("records"),
    id_field="person_id",
    embed_template="{first_name} {last_name}, {occupation}",
    metadata_fields=["age", "street"],
)

# Link 1900 census records
matches = index.query(
    records=census_1900.to_dict("records"),
    embed_template="{first_name} {last_name}, {occupation}",
    top_k=5,
    threshold=0.7,
)

matches.to_csv("census_matches.csv", index=False)
```

### Occupation Deduplication

```python
from semantic_linker import SimilarityIndex
import pandas as pd

# Load occupations
occs = pd.read_csv("occupations.csv")

# Create index
index = SimilarityIndex.create(
    name="occupations",
    persist_path="./occupation_index",
)

# Add with multi-field template
index.add(
    records=occs.to_dict("records"),
    id_field="occupation_id",
    embed_template="German: {german_name} | English: {english_translation} | Description: {description}",
    metadata_fields=["hisco_code", "category"],
)

# Find duplicates
dupes = index.find_duplicates(threshold=0.9)
print(dupes)
```

## API Reference

### SimilarityIndex

#### Class Methods

- `SimilarityIndex.create(name, persist_path)` - Create a new index
- `SimilarityIndex.load(persist_path, name=None)` - Load an existing index

#### Properties

- `count` - Number of records in the index
- `embedded_ids` - List of all record IDs
- `name` - Collection name
- `persist_path` - Storage location (None for ephemeral)

#### Methods

- `add(records, id_field, embed_template, metadata_fields=[], force_batch=False)` - Add records
- `query(records, embed_template, top_k=5, threshold=None, force_batch=False)` - Query for matches
- `find_duplicates(threshold=0.9, exclude_self=True)` - Find duplicate pairs
- `delete(ids)` - Delete records by ID
- `clear()` - Remove all records

### Convenience Functions

- `match(queries, targets, top_k=5, threshold=None, embed_template=None, id_field="id")` - Quick matching
- `dedupe(texts, threshold=0.9, embed_template=None, id_field="id")` - Quick deduplication

## Output Format

All query methods return a pandas DataFrame:

**Query Results:**
| Column | Description |
|--------|-------------|
| query_id | ID of the query record |
| query_text | Formatted query text |
| target_id | ID of the matched record |
| target_text | Formatted target text |
| similarity | Cosine similarity (0-1) |
| rank | Match rank (1-indexed) |
| [metadata] | Any stored metadata fields |

**Duplicate Results:**
| Column | Description |
|--------|-------------|
| id_1 | First record ID |
| text_1 | First record text |
| id_2 | Second record ID |
| text_2 | Second record text |
| similarity | Cosine similarity |

## Environment Setup

Set your Google API key:

```bash
export GEMINI_API_KEY="your-api-key"
```

## Dependencies

- `chromadb` - Vector storage
- `gemini-batch` - Batch embedding API (50% cost savings)
- `google-genai` - Google Generative AI client
- `pandas` - DataFrame output
- `numpy`, `scikit-learn` - Similarity computations

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run unit tests (mocked, no API key needed)
uv run pytest tests/ --ignore=tests/test_integration.py

# Run integration tests (requires GOOGLE_API_KEY)
export GEMINI_API_KEY="your-api-key"
uv run pytest tests/test_integration.py -v

# Run all tests
uv run pytest tests/ -v
```

## License

MIT
