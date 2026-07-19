import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://veloxa-frontend.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("veloxa-inventory")

with open("veloxa_enhanced_catalog.json", "r") as f:
    catalog = json.load(f).get("catalog", [])


def retrieve_relevant_shoes(query: str) -> list:
    query_emb = client.models.embed_content(model="gemini-embedding-001", contents=query)
    search_results = index.query(vector=query_emb.embeddings[0].values, top_k=4, include_metadata=True)
    matched_ids = [int(match["id"]) for match in search_results["matches"]]
    return [shoe for shoe in catalog if shoe["id"] in matched_ids]


@app.get("/")
def read_root():
    return {"message": "Veloxa backend is running"}


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    relevant_shoes = retrieve_relevant_shoes(request.message)

    system_instruction = f"""
    You are the Veloxa AI Concierge, a helpful shoe-shopping assistant.
    RETRIEVED INVENTORY: {json.dumps(relevant_shoes)}

    Only recommend items from RETRIEVED INVENTORY above. If nothing there fits, say so honestly rather than inventing a product. Respond briefly and conversationally.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{system_instruction}\n\nUSER: {request.message}",
    )
    return {"reply": response.text}