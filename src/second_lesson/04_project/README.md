# Section project: Semantic search

## What will be built:

We’ll take detailed movie descriptions and apply the chunking strategies you learned earlier, embed those chunks using sentence transformers, and store them in Qdrant with rich metadata. The result is a search engine that understands themes, moods, and concepts.

This project synthesizes everything from today: points and vectors, distance metrics, payloads, chunking strategies, and embedding models. By the end, you’ll have a working system that can find movies by plot, theme, or emotional resonance.

A semantic search engine that can:

- Understand meaning: Search for “time travel and family relationships” and find Interstellar
- Compare chunking strategies: See how fixed-size, sentence-based, and semantic chunking affect search quality
- Filter intelligently: Combine semantic search with metadata filters (year, genre, rating)
- Handle constraints: Process long movie descriptions that exceed embedding model token limit
- Group results: Avoid duplicate movies when multiple chunks match your query

### Step 1: Understanding the challenge

Our dataset consists of 13 science fiction movies with detailed, literary descriptions. Here’s the challenge: each description contains 240-460 tokens, but our embedding model (all-MiniLM-L6-v2) can only embed 256 tokens or less.

### Step 2: The Three-Vector Experiment

Here’s what makes this demo unique: we’ll create three different vector spaces in a single collection, each representing a different chunking strategy. This lets us directly compare how chunking affects search quality.

Side note: Creating three different vector spaces in a single collection is almost as expensive as having one collection per vector space. We do it here purely for comparison convenienc

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# Initialize components
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory for demo: NO HNSW built -> queries are a full scan.
client = QdrantClient(":memory:")

# For ANN/HNSW:
# client = QdrantClient(url="http://localhost:6333")

# Create collection with three named vectors
client.create_collection(
    collection_name='movie_search',
    vectors_config={
        'fixed': models.VectorParams(size=384, distance=models.Distance.COSINE),
        'sentence': models.VectorParams(size=384, distance=models.Distance.COSINE),
        'semantic': models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
)
```

- Fixed: 40 tokens per chunk
- Sentence: sentence-aware chunks with overlap
- Semantic: Meaning-aware

### Step 3: Implementing the Chunking Strategies

Here’s where the chunking concepts from earlier lessons come alive. We’ll implement three different approaches and see how they perform

### Step 4: Processing and Uploading the Data

For each movie description, we apply all three chunking strategies, embed the resulting chunks, and store them with their respective vector names

### Step 5: Comparing Search Results

Now comes the fascinating part: testing how different chunking strategies affect search quality. Let’s create a helper function to compare results

### Step 6: Advanced Features

- Filtering by Metadata
- Grouping Results to Avoid Duplicates
