import logging
from uuid import uuid4

import httpx
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from .constants import BASE_URL

logging.basicConfig(level=logging.INFO)

model = SentenceTransformer("all-MiniLM-L6-v2")


def fixed_chunking(data: str) -> list[str]:
    """
    Fixed size chunking
    """
    words = data.split()
    chunks = []
    for i in range(0, len(words), 20):
        chunks.append(" ".join(words[i : i + 20]))

    return chunks


def sentence_chunking(data: str) -> list[str]:
    """
    Sentence chunking function
    """
    sentences = sent_tokenize(data)
    chunks, buffer, length = [], [], 0

    for sent in sentences:
        count = len(sent.split())
        if length + count > 20:
            chunks.append(" ".join(buffer))
            buffer, length = [], 0
        buffer.append(sent)
        length += count

    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def semantic_chunking(text: str, similarity_threshold=0.5) -> list[str]:
    """
    Semantic chunking function
    """
    sentences = text.split(".")
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
        )

        if similarity < similarity_threshold:
            chunks.append(". ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(". ".join(current_chunk))
    return chunks


def encode_data(chunks: list[str]):
    """
    Turns chunks into embeddings
    ARGS:
        chunks: list of string chunks to embed
    RETURS:
        list[float]: list of embeddings
    """
    return model.encode(chunks)


def get_animes_by_season(year: int, season: str) -> dict:
    """
    Function responsible for retrieving from api anime synopsis
    ARGS:
        year: int -> year to choose seasons
        season: str -> year season of choice
    Returns:
        Dict[str, Dict[str, str | int]] -> keys being vectors from collection
    """
    url = BASE_URL + f"seasons/{year}/{season.lower()}"

    logging.info(f"Getting from:{url}")

    with httpx.Client() as client:
        response = client.get(url, params={"filter": "tv", "limit": 20})

    if response.status_code != 200:
        logging.info("Failed to get data")
        return {}

    raw_data = response.json().get("data", [])
    logging.info(f"Length of data:{len(raw_data)}")
    data_dict = {}

    for item in raw_data:
        title = None
        id = item.get("mal_id", 0)

        if id:
            genres = []
            for genre in item.get("genres", []):
                genres.append(genre.get("name", ""))

            for item_title in item.get("titles", []):
                if item_title.get("type", "").lower() == "english":
                    title = item_title.get("title", "")

            logging.info(f"Adding: {title}")

            data_dict[id] = {
                "title": title,
                "episodes": item.get("episodes", 0),
                "rating": item.get("rating", "Unknown"),
                "synopsis": item.get("synopsis", "Unknown").replace(
                    "[Written by MAL Rewrite]", ""
                ),
                "genres": genres,
            }

    return data_dict


def parse_data(data: dict[str, dict]) -> dict[str, dict]:
    """
    ARGS:
        data: dict with ids as keys and stores a dict with its content
    RETURNS:
        dict[str, dict]: returns data to be structured to insert at qdrant
    """
    return_dict = {}

    for id, items in data.items():
        synopsis = items.get("synopsis", "")
        fixed_chunk = fixed_chunking(synopsis)
        sentence_chunk = sentence_chunking(synopsis)
        semantic_chunk = semantic_chunking(synopsis)

        encoded_fixed = encode_data(fixed_chunk)
        encoded_sentence = encode_data(sentence_chunk)
        encoded_semantic = encode_data(semantic_chunk)

        payload = {
            "title": items.get("title", ""),
            "episodes": items.get("episodes", 0),
            "rating": items.get("rating", "Unknown"),
            "genres": items.get("genres", []),
            "mal_id": id,
        }

        for fixed, sentence, semantic in zip(
            encoded_fixed, encoded_sentence, encoded_semantic
        ):
            fixed_id = uuid4()
            return_dict[fixed_id] = {
                "id": fixed_id,
                "vector": {"fixed": fixed, "sentence": sentence, "semantic": semantic},
                "payload": payload,
            }

    return return_dict
