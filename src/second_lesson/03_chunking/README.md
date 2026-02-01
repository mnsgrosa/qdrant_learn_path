# Text chunking strategies

Embedding the whole document is a poor strategy, since parts of text can have different meanings for our embedding, <br>
and so it is best advised to break into smaller sections, what we call chunking

## Reasons for not embedding whole documents:

### 1. Context window:

Each model can process a different size of tokens. Most sentence-transformers operate with 512 tokens <br>
while openAI's text-embedding-3-small operates with 8191 and if a document passes this much tokens it <br>
simply drops whatever comes next the last token processed.

### 2. Diluted meaning:

If it fits the context window it still faces the dilution issue, each section carries a meaning as stated <br>
earlier, so we can extract efficiently information for querying we must concetrate what each section carries <br>
the trade off goes by: meaning density -> meaning / size. The higher the better.

### 3. Better querying:

A consequence of better density and sectionized document, here we can add metadata as which section of text <br>
we're storing: title, abstract, conclusion, page number etc... and this makes a better 

## Strategies for chunking:

Here are asomme ways to engineer it

- 1.Fixed-size 
- 2.Sentenced-based
- 3.Paragraph-based
- 4.Sliding window
- 5.Recursive
- 6.Semantic aware

### 1. Fixed-size:

Define a number of tokens per chunk (e.g., 200) with a overlap between sections

```
The HNSW algorithm builds a multi-layer graph where each node represents a vector. The algorithm starts by inserting vectors into the bottom layer and then selectively promotes some to higher layers based on probability. This creates shortcuts that allow for faster traversal during search operations.
```

chunk 1:"The HNSW algorithm builds a multi-layer graph where each"
Chunk 2: “node represents a vector. The algorithm starts by inserting vectors”
Chunk 3: “into the bottom layer and then selectively promotes some to”
Chunk 4: “higher layers based on probability. This creates shortcuts that allow”
Chunk 5: “for faster traversal during search operations."

only the last one doesn't contain 10 words

#### Pros:

- Simple to implement
- Consistent chunk sizes
- Predictable processing

#### Cons:

- Ignores natural language boundaries
- May split mid-sentence or mid-thought
- No semantic awareness

#### Best for:

Documents lacking consistent formatting. inital prototyping

#### Code with overlap of 50 tokens:

```python
def fixed_size_chunk(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

### 2. Sentence-based:

Break into sentences with tokenizer then group sentences into chunks under word count

```
The HNSW algorithm builds a multi-layer graph. Each node represents a vector in the collection. The algorithm creates shortcuts between layers for faster search. This hierarchical structure enables efficient approximate nearest neighbor queries.
```

Sentence-based chunks (respecting sentence boundaries):

Chunk 1: “The HNSW algorithm builds a multi-layer graph. Each node represents a vector in the collection."
Chunk 2: “The algorithm creates shortcuts between layers for faster search. This hierarchical structure enables efficient approximate nearest neighbor queries."

#### code:

```python
from nltk.tokenize import sent_tokenize


def sentence_chunk(text, max_words=150):
    sentences = sent_tokenize(text)
    chunks, buffer, length = [], [], 0

    for sent in sentences:
        count = len(sent.split())
        if length + count > max_words:
            chunks.append(" ".join(buffer))
            buffer, length = [], 0
        buffer.append(sent)
        length += count

    if buffer:
        chunks.append(" ".join(buffer))
    return chunks

```

#### Pros:

- Preserves complete thoughts
- Natural languages boundaries
- Good semantic coherence

#### Cons:

- Irregular chunk lengths
- Sentence size varies significantly
- May not respect topic boundaries

#### Best for:

Rag systems, Q&A applications, general text processing

### 3. Paragraph-based:

Split with document structure in mind

```
Paragraph 1: "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search. It builds a multi-layer structure where each layer contains a subset of the data points."

Paragraph 2: "The algorithm works by creating connections between nearby points in each layer. Higher layers have fewer points but longer connections, creating shortcuts for faster traversal during search operations."

Paragraph 3: "When searching, HNSW starts from the top layer and gradually moves down, using the shortcuts to quickly navigate to the target region before performing a more detailed search in the bottom layer."
```

each chunk is an entire paragraph, generally speaking a natural boundary where ideas tend to cohere keeping related <br>
concepts together <br>

#### Code:

```python
def paragraph_chunk(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]
```

#### Pros:

- Aligns with natural topic boundaries
- Semantically rich by default
- Respects author's organization

#### Cons:

- Unpredictable sizes (single line to whole page)
- May need token limits or fallback splitting
- Depends on clean document structure

#### Best for:
Articles, blogs, documentation, books, emails

### 4. Sliding window:

The name is carried over the algorithms of sliding window where there is information overlap for continuity

```
"HNSW builds a multi-layer graph where each node represents a vector. The algorithm starts by inserting vectors into the bottom layer and then selectively promotes some to higher layers based on probability. This creates shortcuts that allow for faster traversal during search operations."
```

Overlap of 10 words per chunk, 4 words overlap:

Chunk 1: "HNSW builds a multi-layer graph where each node represents a"
Chunk 2: "where each node represents a" "vector. The algorithm starts by inserting vectors"
Chunk 3: "starts by inserting vectors" "into the bottom layer and then selectively promotes"
Chunk 4: "and then selectively promotes" "some to higher layers based on probability. This"

#### Code:

```python
def sliding_window(text, window=200, stride=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words) - window + 1, stride):
        chunk = " ".join(words[i:i + window])
        chunks.append(chunk)
    return chunks
```

#### Pros:

- Keeps context boundaries
- Higher recall potential
- Reduces information loss

#### Cons:

- Storage redundancy (20-50% overhead)
- Increased processing costs
- May return duplicate information

#### Adaption:

(My intuition)
May need a higher threshold of similarity for querying and stronger metadata filtering

#### Best for:

Critical applications where missing info is costly, reranking systems

### 5. Recursive:

Fallback hierarchy of separators when data doesn't follow predictable structure

```
# HNSW Overview\n\nThe HNSW algorithm builds a multi-layer graph.\nEach node represents a vector in the collection.\n\nThe algorithm creates shortcuts between layers for faster search. This hierarchical structure enables efficient approximate nearest neighbor queries.\n\n## Performance Benefits\nHNSW provides logarithmic search complexity.
```

Recursive chunking (tries paragraph breaks first, then sentences, then words):

Chunk 1: # HNSW Overview
Chunk 2: The HNSW algorithm builds a multi-layer graph.
Chunk 3: Each node represents a vector in the collection.
Chunk 4:The algorithm creates shortcuts between layers for faster search.
Chunk 5:This hierarchical structure enables efficient approximate nearest neighbor queries.
Chunk 6:## Performance Benefits\nHNSW provides logarithmic search complexity.

Here the objective is to keep structure it starts looking for paragraphs. If too long it drops down <br>
to sentences. If it overflows again. it cuts to words boundaries.

#### Code:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
)

text = """
Hello, world! More text here. another line.
Hello, world! More text here. another line......
"""
chunks = splitter.split_text(text)
```

#### Pros:

- Adaptable to messy or incosistent input
- Preserves semantic coherence when possible
- Handles various document formats

#### Cons:

- Heuristic-based, may be incosistent
- Complex logic
- May not work perfectly with all content types

#### Best for:

Scraped web content, mixed formats, CMS export

### 6. Semantic-Aware:

Use embeddings to detect meaning shifts and break at topic boundaries, up until the last one was based <br>
on structure here we shift to meaning, here with embedding we find meaning whenever topic or semantic <br>
coherence changes we break the chunk there

```
HNSW is a graph-based algorithm for vector search. It builds hierarchical layers for efficient navigation. The algorithm uses probability to promote nodes between layers. Vector databases like Qdrant implement HNSW for fast similarity search. Machine learning models generate embeddings for text data. These embeddings capture semantic meaning in high-dimensional space.
```

Semantic-aware chunks (splits detected at meaning boundaries):

Topic 1 - HNSW Algorithm: "HNSW is a graph-based algorithm for vector search. It builds hierarchical layers for efficient navigation. The algorithm uses probability to promote nodes between layers."
Topic 2 - Vector Databases: "Vector databases like Qdrant implement HNSW for fast similarity search."
Topic 3 - Machine Learning: "Machine learning models generate embeddings for text data. These embeddings capture semantic meaning in high-dimensional space."

Here we don't care about sentence count or fixed token limits. But for natural meaning. If a definition <br>
needs need multiple sentences we keep them together. Providing better retrieval due to coherent concepts

#### Code:

- 1.Embed sentences or small segments
- 2.Calculate similarity between consecutive segments
- 3.Identify topic transitions where similarity drops
- 4.Split at coherence boundaries

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunking(text, similarity_threshold=0.5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = text.split('.')
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Calculate cosine similarity between consecutive sentences
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        
        if similarity < similarity_threshold:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    chunks.append('. '.join(current_chunk))
    return chunks
```

Here the trade-off is computational cost. Embedding the full document upfront to decide where to split <br>
so it's slower and expensive but each chunk is coherent

#### Pros:

- High semantic precision
- Each chunk carries coherent ideas
- Optimal for complex documents

#### Cons:

- Computationally expensive (embeds the whole document)
- Requires additional model inference
- Slower processing pipeline

#### Best for:

Legal documents, research papers, critical applications requiring high precision

## Adding meaning with metadata:

As we've seen metadata is a powerfull tool, it provides a even better querying strategy telling where the <br>
document comes from, size of chunk, section title, document type etc... <br>

Example of a good metadata (payload):

```json
{
  "document_id": "collection-config-guide",
  "document_title": "What is a Vector Database",
  "section_title": "What Is a Vector",
  "chunk_index": 7,
  "chunk_count": 15,
  "url": "https://qdrant.tech/documentation/concepts/collections/",
  "tags": ["qdrant", "vector search", "point", "vector", "payload"],
  "source_type": "documentation", 
  "created_at": "2025-01-15T10:00:00Z",
  "content": "There are three key elements that define a vector in vector search: the ID, the dimensions, and the payload. These components work together to represent a vector effectively within the system...",
  "word_count": 45,
  "char_count": 287
}
```

## What metadata enables:

*DISCLAIMER:* Filterable fiels must be indexed using payload index

### 1.Filtered search (Exact match)

Perfect for categorical data:

```python
from qdrant_client import models

# Only show results from a specific article
filter = models.Filter(
    must=[
        models.FieldCondition(
            key="document_id", match=models.MatchValue(value="collection-config-guide")
        )
    ]
)
```

### 2.Hybrid search with text filtering (full-text search):

combines vectors with traditional keyword search:

```python
# Find vectors that also contain the keyword "HNSW" in their content
filter = models.Filter(
    must=[
        models.FieldCondition(
            key="content", # The field with the full-text index
            match=models.MatchText(text="HNSW algorithm")
        )
    ]
)
```

### 3.Grouped results:

```python
group_by = "document_id"
```

### 4.Rich result display:

- Original content with source attribution
- Section context for better understanding
- Direct links to full documents
- Creation timestamps for freshness

### 5.Permission control:

```python
# Filter by user permissions
filter = models.Filter(
    must=[
        models.FieldCondition(
            key="access_level", match=models.MatchValue(value="public")
        )
    ]
)
```

### Search with metadata:

```python
def search_with_filters(query, document_type=None, date_range=None):
    """Search with metadata filtering"""

    # Build filter conditions
    filter_conditions = []

    if document_type:
        filter_conditions.append(
            models.FieldCondition(
                key="source_type", match=models.MatchValue(value=document_type)
            )
        )

    if date_range:
        filter_conditions.append(
            models.FieldCondition(
                key="created_at",
                range=models.Range(gte=date_range["start"], lte=date_range["end"]),
            )
        )

    # Execute search
    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    results = client.query_points(
        collection_name="documents",
        query=generate_embedding(query),
        query_filter=query_filter,
        limit=5,
    )

    return results
```

### Performance considerations:

#### 1.Token efficiency

Consider embedding model's token limits:

- Check always model documentation for context window
- Leave buffer space for special tokens and formatting

#### 2.Overlap recommendation:

- 10-20% overlap: Good balance for most applications
- 25-50% overlap: High-recall scenarios where missing info is critical
- No overlap: Storage/compute costs are primary concerns

## Takeaways:

- Chunking strategy directly impacs search quality
- Smaller and focused chunks are better than whole document
- Metadata is crucial
- Check which trade-offs best fit your case
- Consider computational costs -> smantic chunking is powerful but expensive
- Overlap helps preserve context but increases storage requirements
