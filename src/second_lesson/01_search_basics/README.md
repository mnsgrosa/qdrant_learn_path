## Points:

At qdrant the point consists fo 3 elements: id, vector and payload <br>

## Vector types in qdrant:

### Dense vectors:

Typical vectors generate by most of neural networks often called embeddings

### Sparse vectors:

Mathematically the same as dense vectors, but containing many zeros. Those are used for optimized <br>
storage representation and have different shapes of vectors

#### Representation of sparse vectors:

the representation used follows the rule of list of (index, value) pairs: <br>
index: integer position of non-zero values<br>
value: floating point number<br>

```
[(6, 1.0), (7, 2.0)]
```

as Qdrant JSON format:

```
{
"indices":[6,7],
"values":[1.0,2.0]
}
```

both indices and values must be the same size and indices must be unique and no need to sort <br>
for qdrant as qdrant deals with it

### Multivectors:

Basically a tensor: a vector of vectors :)

```
"vector": [
   [-0.013,  0.020, -0.007, -0.111],
   [-0.030, -0.055,  0.001,  0.072],
   [-0.041,  0.014, -0.032, -0.062],
   # ...
]
```
### Named Vectors:

We can also place multiple vector types for a single qdrant point and we must specify while configurating

```
client.create_collection(
    collection_name="{collection_name}",
    vectors_config={
        "image": models.VectorParams(size=4, distance=models.Distance.DOT),
        "text": models.VectorParams(size=5, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={"text-sparse": models.SparseVectorParams()},
)

client.upsert(
    collection_name="{collection_name}",
    points=[
        models.PointStruct(
            id=1,
            vector={
                "image": [0.9, 0.1, 0.1, 0.2],
                "text": [0.4, 0.7, 0.1, 0.8, 0.1],
                "text-sparse": {
                    "indices": [1, 3, 5, 7],
                    "values": [0.1, 0.2, 0.3, 0.4],
                },
            },
        ),
    ],
)
```

Here we can check that one point has two vectors associated, image(dense), text(dense) and text-sparse(sparsed)

## Commonly used dimensions and models

all-miniLm-l6-v2 -> 384 -> prototyping
bge-base-en-v1.5 -> 768 -> baseline for rag
openai-text-embedding-3-small -> 1536 -> commercial model for semantic search
text-embedding-3-large -> 3072 -> maximun detail commecial model larg scale and high accuracy rag

Notice that all those double the size 384 * 2 = 768 * 2 = 1536 * 2 = 3072

## Comon embedding sources:

### 1. FastEmbed by Qdrant (on-premise)

- cpu friendly running on ONNX makin it 50% faster than pytorch based models
- Uses by deffault bge-base-en-v1.5 for its size ~67MB

Qdrant lets yus use any compatibnle model

#### When should you use FastEmbed

- On-premise execution for privacy
- High speed cpu without pytorch
- Scalable but low cost embedding generation solution

### 2. Managed and integrated: Cloud providers

- Qdrant cloud inference sending images or text in a single request
- OpenAI and Anthropic cost based

#### When should you use:

- Ease of use and offload model management and infrastructure scaling
- Need access to latest commecial models with minimal setup
- can accept API costs and latency for high-quality embeddings

### 3. On-premise, customizable: Open Source Models

libraries sentence transformers from hugging face giving maximun flexibility and control

#### When should you use:

- Fine-tune needed
- Full control over model architecture and deplyoment env
- Has available GPU

## Payloads (Metadata)

Filter and refiner of querying vectors storing things as: date, price, descriptions, ratings and even complex structures

## Payload types:

- keyword: exact string matching
- integer: 64-bit for numerical filtering
- float: 64-bit for prices, ratings etc...
- bool: true/false
- geo: latitude/longitude pairs for location based queries
- datetime: RFC-3339 firnat
- UUID

### Data structures:

- arrays
- json

## Filtering logic:

Complex queries can be created nesting filters using the following clauses:

- must -> all conditions must be satisfied (AND logic)
- should -> at least one (OR logic)
- must_not -> None must be met (NOT logic)

Example of query

```
models.Filter(
    should=[
        models.Filter(must=[
            models.FieldCondition(key="category", match=models.MatchValue(value="electronics")),
            models.FieldCondition(key="price", range=models.Range(lt=200))
        ]),
        models.Filter(must=[
            models.FieldCondition(key="category", match=models.MatchValue(value="books")),
            models.FieldCondition(key="rating", range=models.Range(gte=4.0))
        ])
    ]
)
```

- match: exact value
- range
- geo: location based
- full text: specific words or phrases whith a text field
- Nested inside of arrays of objects

### filtering capabilities reference:

- match: exact
- match any: or logic
- match except: not in logic
- range: numerical ranges -> "range":{"gte":50, "lte":200}
- datetime range: date range -> "range":{"gt":2023-01-01T00:00:00Z"}
- full text: substring match -> "match":{"text":"amazing services"}
- geospatial: location based -> "geo_radius": {"center":{...}. "radius":10000}
- nested: array object -> "nested":{"key": "reviews", "filter": {...}}
- has id: specific id -> "has_id":[1, 5, 10]
- is empty: missing fields -> "is_empty":{"key":"some_metadata_key"}
- is null: null values -> "is_null":{"key":"some_metadata_key"}
- velues count: array length -> "values_count":{"gt":2}

### Advanced filtering

object:

```
{
  "id": 1,
  "product": "Laptop",
  "reviews": [
    {"user": "alice", "rating": 5, "verified": true},
    {"user": "bob", "rating": 3, "verified": false}
  ]
}
```

filter:

```
models.Filter(
    must=[
        models.NestedCondition(
            nested=models.Nested(
                key="reviews",
                filter=models.Filter(must=[
                    models.FieldCondition(key="rating", match=models.MatchValue(value=5)),
                    models.FieldCondition(key="verified", match=models.MatchValue(value=True))
                ])
            )
        )
    ]
)
```

### performance optimization

To maximize filtering, create payload indexes for frequently filtered fields

Examples:
```
# Index frequently filtered fields
client.create_payload_index(
    collection_name="{collection_name}",
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

# For multi-tenant applications, mark tenant fields
client.create_payload_index(
    collection_name="{collection_name}",
    field_name="tenant_id",
    field_schema=models.KeywordIndexParams(type="keyword", is_tenant=True),
)

```

whenever there are too many filters qdrant may bypass vector indexing entirely and use payload indexes for faster results