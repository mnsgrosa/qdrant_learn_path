# Why indexing your vectors

While using a vector database whenever itends to search for a similar vector we must search clustering them <br>
so if a db has millions to billions entries using purely similarity will get really slow but in qdrant it is <br>
used HNSW algorithm for faster querying.

## HNSW

HNSW lays out a structure that conects the vector points as graphs and let's us filter them as we desire for <br>
a more efficient search time, if it's known that you want to query for a specific document why search all vectors? <br>

## Configuring HNSW

We can fine tune ou graph traversing with some hyperparameters for our indexing using m, ef_construct and hnsw_ef:

### Graph connectivity: m

This parameter controls the maximum number of connections per node in the graph

- Higher m means denser graph with more neighbors improving search accuracy and increases memory usage and indexing time
- lower m means sparser graph, less memory usage and sped up insertion but less accurate
- typically goes by 8 to 64

### Build toughness: ef_construct

How many candidates are checked while inserting a vector

- Higher values means more neighbors evaluated, resulting in a more comphensive and accurate graph, indexing slower an higher computing
- lower values means sped up insertion but graph end up with less optimal connections, which can impact search accuracy
- values 100 to 500 are commonly used higher values are better for complex data

### Search thoroughness: hnsw_ef

number of candidates evaluated during search query

- Higher values translate to more accurate research but increases query time
- Lower values less accuracy and speeds up
- range 50 to 200+ depending on latency targer

## Optimal settings:

### High speed retrieval:

Lower *m* and *hnsw_ef* and *ef_construct* high enough for acceptable recall

### Maximum recall:

Raise all parameters and accept slower queries and builds

### Tight ram:

Reduce *m* keep *ef_construct* construct high enough to avoid poor links

## HNSW in action

While brute force grows O(N) HNSW grows roughly O(log(n)) makin milion scale dataset searchable in seconds <br>
it is filter aware through indexing allowing fast searches under structured conditions. This avoids costly full scans when filtering by<br>
metadata. Latly it suports real time updates with high recall, fits semantic search and recommendation systems, scales from thousands <br>
to bilions of vectors

## Practical configuration:

### Production:

```python
client.create_collection(
    collection_name="production_vectors",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfigDiff(
            m=16,  # Balanced connections (default)
            ef_construct=200,  # Good build quality (default)
            full_scan_threshold=10000,  # Use brute force below this size (default)
        ),
    ),
)
```

### Development or testing: faster builds

```python
client.create_collection(
    collection_name="dev_vectors",
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfigDiff(
            m=8,  # Fewer connections
            ef_construct=100,  # Faster builds
            full_scan_threshold=10000,  # Use brute force below this size (default)
        ),
    ),
)
```

## Performance benchmarking:

First we create and upload some toy data to a new collection

```python
import time
from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

collection_name = "my_collection"

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)


# Development / testing: faster builds
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=4,
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfigDiff(
            m=8,  # Fewer connections
            ef_construct=100,  # Faster builds
            full_scan_threshold=100,  # Use brute force below this size (default)
        ),
    ),
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=100,  # Use brute force below this size (default)
    ),
)

# upload data
import random

points = []
for i in range(20000):
    points.append(
        models.PointStruct(id=i, vector=[random.random() for _ in range(4)], payload={})
    )
client.upload_points(
    collection_name=collection_name,
    points=points,
)
```

### Search Performance:

```python
def benchmark_search_performance(collection_name, test_queries, ef_values):
    """Compare latency across hnsw_ef values"""

    results = {}
    for hnsw_ef in ef_values:
        start_time = time.time()
        for query in test_queries:
            client.query_points(
                collection_name=collection_name,
                query=query,
                limit=10,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            )

        avg_time = (time.time() - start_time) / len(test_queries)
        results[hnsw_ef] = avg_time
        print(f"hnsw_ef={hnsw_ef}: {avg_time:.3f}s per query")

    return results


# Test different hnsw_ef values
test_queries = [
    [30, 60, 90, 120],
    [150, 180, 210, 240],
    [270, 300, 330, 360],
    [390, 420, 450, 480],
    [510, 540, 570, 600],
]

ef_values = [32, 64, 128, 256]
performance = benchmark_search_performance(collection_name, test_queries, ef_values)
```

### Inspecting performance and index use:

The get_collection method returns current stats and configuration from collections like points_count, indexed_vectors_count or hnsw_config. It also lists payload_schema for payload indexes created

#### If queries fell slow:

- Check if filter fields have payload indexes
- If the payload was set before building HNSW graph with setting m>0
- If the payload was set before HNSW building graph (if m changes from 0 to higher number it builds automatically)
- If *hnsw_config.full_scan_threshold* is too high

Here how to inspect:

```python
# Inspect collection status
info = client.get_collection(collection_name)

vectors_per_point = 1  # set per your vectors_config
vectors_count = info.points_count * vectors_per_point

print(f"Collection status: {info.status}") 
print(f"Total points: {info.points_count}")
print(f"Indexed vectors: {info.indexed_vectors_count}")

if vectors_count:
    proportion_unindexed = 1 - (info.indexed_vectors_count / vectors_count)
else:
    proportion_unindexed = 0

print(f"Proportion unindexed: {proportion_unindexed:.2%}")

if info.status == models.CollectionStatus.GREEN:
    print("\n✅ Collection is indexed and ready!")
elif info.status == models.CollectionStatus.YELLOW:
    print("\n⚠️ Collection is still being indexed (optimizing).")
else:
    print(f"\n❌ Collection status is {info.status}.")
```

## When not to use HNSW:

### Small collections
generally lesser than 10000 vectors, brute force in this case is faster and uses less RAM than building HNSW

### Exact search requirements

HNSW is apporixmate, if needed exact results, use brute force

### Extreme memory constraints

for very tight RAM budget:

- Lower m: HNSW memory follows O(m * vector_count)
- Vector scalar quantization: quantization often cuts RAM ~4x
- Vector binary quantization: compresses to 1-bit per dimension and can cut RAM by large factors
- On disk storage: set on_disk=True for vector and HNSW index to use mmap files only the most visited are cached in ram
- Disable HNSW for reranking embeddings: For multi-vectors reranking is too costly
