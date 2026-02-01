import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from utils.helper_functions import get_animes_by_season, parse_data

load_dotenv()


class ClientManager:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=300,
        )
        self.collection_name = "anime_synopsis"
        self.check_collection()
        self.data = {}
        self.parsed_data = {}

    def check_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "fixed": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    ),
                    "sentence": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    ),
                    "semantic": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    ),
                },
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="title",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def store_synopsis(self, year, season):
        self.data = get_animes_by_season(year, season)
        return self

    def parse_synopsis(self):
        if self.data:
            self.parsed_data = parse_data(self.data)
        return None

    def create_structs(self):
        return [models.PointStruct(**self.parsed_data[id]) for id in self.parsed_data]

    def upsert_date(self):
        if self.parsed_data is not None:
            self.client.upsert(
                collection_name=self.collection_name, points=self.create_structs()
            )
        return None


if __name__ == "__main__":
    client = ClientManager()
    client.store_synopsis(2025, "summer").parse_synopsis()
    client.upsert_date()
