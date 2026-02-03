# Combining vector search and filtering

In real world is ideal to use filters. With this there are some challenges that qdrant solves

## The callenge: filters break graph connectivity

Let's check how range filters can show us this callenge:<br>
Imagine we're searching for computer prices and we set to prices lower than 1000 and categorical filter for laptop. This may break paths in our graph since we may not have node connections sufficient for a traversal, not because of similarity but rather because of lack of paths connecting filtered out

### Naive approaches and their problems:

#### Post-filtering:

Get top k most similar ones and filter out. This one has an issue, if the best match wasn't in that top k, you won't retrieve it, you will waste compute power and lose recall because of relevant points neve were retrieved

#### Pre-filtering:

Too restrictive filters fragment HNSW, breaking connectivity and traversal becomes inefficient or impossible

## Qdrant solution: Filterable HNSW:

Qdrant creates additional edges to maintain connectivty under filtering. There is a subgraph for each payload value, then merges it back to the original graph

## Query planner: Adaptive strategy

This happens on query time using a query planner it happens per segmend and is based on filter cardinality

- Filter matches many points: HNSW skips nodes without match traversal avoiding pre filtering
- Filter matches few points: If it matches a really smal proportion qdrant skips HNSW and fall for a full scan

## Payload indexing:

If it is planned to add indexes do as building the collection for is it known it is compute heavy to rebuild HNSW

```python
from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# For Colab:
# from google.colab import userdata
# client = QdrantClient(url=userdata.get("QDRANT_URL"), api_key=userdata.get("QDRANT_API_KEY"))

collection_name = "store"
vector_size = 768

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
    ),
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=100,
    ),
)

# Index frequently filtered fields
client.create_payload_index(
    collection_name=collection_name,
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="price",
    field_schema=models.PayloadSchemaType.FLOAT,
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="brand",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
```

Upload

```python
# Upload data
import random

points = []
for i in range(1000):
    points.append(
        models.PointStruct(
            id=i,
            vector=[random.random() for _ in range(vector_size)],
            payload={
                "category": random.choice(["laptop", "phone", "tablet"]),
                "price": random.randint(0, 1000),
                "brand": random.choice(
                    ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "Samsung"]
                ),
            },
        )
    )
client.upload_points(
    collection_name=collection_name,
    points=points,
)
```

## Memory considerations:

Additional indexes add up memory consumpiton, so only index fields used in filtering conditions

## Practical implementation:

```python
# Create filter combining multiple conditions
filter_conditions = models.Filter(
    must=[
        models.FieldCondition(key="category", match=models.MatchValue(value="laptop")),
        models.FieldCondition(key="price", range=models.Range(lte=1000)),
        models.FieldCondition(key="brand", match=models.MatchAny(any=["Apple", "Dell", "HP"])),
    ]
)

query_vector = [random.random() for _ in range(vector_size)]

# Execute filtered search
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=filter_conditions,
    limit=10,
    search_params=models.SearchParams(hnsw_ef=128),
)

```

## Query planner decision matrix:

- higher cardinality means hnsw with node skipping and should be usen to filter matches many points
- very low means full scan over candidates and tiny result set

## Performance optimization tips:

- Index early
- Index right fields
- Test filter combinations
- Tune threshold: Adjust full_scan_threshold based on data distribuition patterns
- Measure real performance
