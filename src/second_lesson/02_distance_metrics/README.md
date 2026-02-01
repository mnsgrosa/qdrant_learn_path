# Distance metrics

After storing the vector it is performed a KNN search to retrieve to retrieve semantically similar items based on how close they are in space

## Quick rule of thumb

Most of the time it isn't needed to design you own metric from scratch<br>

- third party embedding models
  - Use the metric recommended in model docs often cosine or dot
  - create your qdrant collection with that metric
-  If the model docs doesn't tell
  - Pick cosine in qdrant
  - qdrant normalize it in cosine collections

## When to use each:

### Cosine: NLP embeddings, semantic search, document retrieval
Direction (orientation), magnitude is ignored <br>
how score is given 

- 1 - same direction
- 0 - orthogonal vectors
- -1 - opposite directions

### Dot product: Recommenders, matrix factorization, ranking
both magnitude and direction


#### When to use:

- Recommenders: If a user vector represents preferences, larger values mean higher relevance
- Asymmetric search: when one vector is short and the document vector is long, and that length implies more information or higher confidence

### Euclidean distance (L2): Spatial data, anomaly detection, clustering
absolute distance (magnitude). <br>
sensitive to scale

### Manhatan distance: Sparse data, robust outlier handling
Grid based distance (sum of absolute differences)


## Experiment tracking

Qdrant lets you use different metrics per named vector making a/b testing to check metrics on your data easier
