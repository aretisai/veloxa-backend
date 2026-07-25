import os
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient, models
import psycopg2

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

COLLECTION_NAME = "veloxa-inventory-qdrant"
EMBEDDING_DIM = 3072  # gemini-embedding-001's default output size

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()
cur.execute("""
    SELECT id, model, category, gender, colors_available, performance_specs, final_price
    FROM shoes ORDER BY id
""")
shoes = cur.fetchall()
cur.close()
conn.close()
print(f"Loaded {len(shoes)} shoes from PostgreSQL.")

try:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}'.")
except Exception as e:
    print(f"Collection '{COLLECTION_NAME}' already exists or creation failed ({e}) - continuing; matching IDs will be overwritten.")

points = []
for (sid, model, category, gender, colors, specs, price) in shoes:
    specs = specs or {}
    text = (
        f"{model} - {category} shoe for {gender}. "
        f"Colors: {', '.join(colors)}. "
        f"Support: {specs.get('support_type', 'standard')}. "
        f"Cushioning: {specs.get('cushioning_level', 'standard')}. "
        f"Price: ${price}."
    )
    embedding = client.models.embed_content(model="gemini-embedding-001", contents=text)
    vector = embedding.embeddings[0].values
    points.append(models.PointStruct(id=sid, vector=vector, payload={"model": model}))
    print(f"  Embedded #{sid}: {model}")

qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

count = qdrant.count(collection_name=COLLECTION_NAME, exact=True)
print(f"\nDone. Collection '{COLLECTION_NAME}' now has {count.count} points.")