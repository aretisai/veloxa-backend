import os
import json
import time
import re
import uuid
import base64
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone
import cohere
import psycopg2
from langfuse import observe, get_client, propagate_attributes

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
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

store_policies = {
    "shipping": "Free standard shipping on orders over $150. Expedited shipping is $25.",
    "returns": "30-day trial period. Take them for a run!",
    "exchanges": "Free size and color exchanges within 30 days.",
}


# ==========================================
# CATALOG: PostgreSQL, with local JSON fallback
# ==========================================
def load_catalog_from_db() -> list:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()

    cur.execute("""
        SELECT id, model, category, gender, price, final_price, cost,
               gross_margin, gross_margin_pct, financial_tier,
               colors_available, performance_specs
        FROM shoes ORDER BY id
    """)
    shoe_rows = cur.fetchall()

    cur.execute("SELECT shoe_id, color, size, stock, image FROM inventory ORDER BY shoe_id")
    inventory_rows = cur.fetchall()
    cur.close()
    conn.close()

    inventory_by_shoe: dict = {}
    for shoe_id, color, size, stock, image in inventory_rows:
        inventory_by_shoe.setdefault(shoe_id, []).append(
            {"color": color, "size": size, "stock": stock, "image": image}
        )

    result = []
    for (sid, model, category, gender, price, final_price, cost,
         gross_margin, gross_margin_pct, financial_tier,
         colors_available, performance_specs) in shoe_rows:
        result.append({
            "id": sid,
            "model": model,
            "category": category,
            "gender": gender,
            "price": int(price),
            "finalPrice": int(final_price),
            "cost": int(cost),
            "gross_margin": float(gross_margin) if gross_margin is not None else None,
            "gross_margin_pct": float(gross_margin_pct) if gross_margin_pct is not None else None,
            "financial_tier": financial_tier,
            "colors_available": colors_available,
            "performance_specs": performance_specs,
            "inventory": inventory_by_shoe.get(sid, []),
        })
    return result


try:
    catalog = load_catalog_from_db()
    print(f"Loaded {len(catalog)} shoes from PostgreSQL.")
except Exception as e:
    print(f"Postgres load failed ({e}) - falling back to local JSON.")
    with open("veloxa_enhanced_catalog.json", "r") as f:
        catalog = json.load(f).get("catalog", [])


# ==========================================
# GOVERNANCE: PII + HITL
# ==========================================
@observe(as_type="span", name="PII_Scrubber")
def scrub_pii(text: str, trace: list) -> str:
    trace.append(f"[{time.strftime('%H:%M:%S')}] Security: Scrubbing PII...")
    scrubbed = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CC]", text)
    scrubbed = re.sub(r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b", "[REDACTED_PHONE]", scrubbed)
    if scrubbed != text:
        trace.append(f"[{time.strftime('%H:%M:%S')}] Security: PII detected and redacted.")
    return scrubbed


@observe(as_type="span", name="Intent_Router")
def check_hitl_escalation(text: str, trace: list) -> bool:
    trace.append(f"[{time.strftime('%H:%M:%S')}] Router: Evaluating intent for HITL escalation...")
    keywords = ["refund", "fraud", "lawsuit", "sue", "manager"]
    if any(k in text.lower() for k in keywords):
        trace.append(f"[{time.strftime('%H:%M:%S')}] Router: High-risk keyword detected. Escalating to HITL.")
        return True
    return False


# ==========================================
# RETRIEVAL (Pinecone + Cohere rerank)
# ==========================================
def build_search_query(safe_text: str, history: list) -> str:
    recent = " ".join(msg["text"] for msg in history[-2:])
    return f"{recent} {safe_text}".strip()


@observe(as_type="span", name="Vector_Retrieval")
def retrieve_relevant_shoes(query: str, trace: list) -> list:
    trace.append(f"[{time.strftime('%H:%M:%S')}] RAG: Querying Vector DB...")
    query_emb = client.models.embed_content(model="gemini-embedding-001", contents=query)
    search_results = index.query(vector=query_emb.embeddings[0].values, top_k=15, include_metadata=True)

    matched_ids = [int(match["id"]) for match in search_results["matches"]]
    candidates = [shoe for shoe in catalog if shoe["id"] in matched_ids]
    if not candidates:
        return []

    documents = [
        f"{shoe['model']} - {shoe['category']} - ${shoe['finalPrice']} - Colors: {', '.join(shoe['colors_available'])}"
        for shoe in candidates
    ]
    rerank_response = co.rerank(
        model="rerank-v4.0-fast",
        query=query,
        documents=documents,
        top_n=min(4, len(documents)),
    )
    trace.append(f"[{time.strftime('%H:%M:%S')}] RAG: Retrieved and reranked {len(rerank_response.results)} items.")
    return [candidates[r.index] for r in rerank_response.results]


def find_shoe_by_id(shoe_id: int) -> dict | None:
    return next((s for s in catalog if s["id"] == shoe_id), None)


# ==========================================
# TOOL CALLING (scoped per-request, not global)
# ==========================================
def make_cart_tools(trace: list, cart_actions: list, cart_removals: list, cart_cleared: list):
    @observe(as_type="span", name="Tool_Execution")
    def add_to_cart(shoe_id: int) -> str:
        """Add an item to cart, identified ONLY by its numeric id from RETRIEVED INVENTORY.
        Name and price are looked up from PostgreSQL directly - never taken from the model."""
        matched = find_shoe_by_id(shoe_id)
        if not matched:
            trace.append(f"[{time.strftime('%H:%M:%S')}] Error: add_to_cart called with unknown shoe_id {shoe_id}.")
            return f"Error: No shoe with id {shoe_id} exists. Ask the user to clarify which item they mean."

        final_name = matched["model"]
        final_price = matched["finalPrice"]
        cart_actions.append({"name": final_name, "price": final_price})
        trace.append(f"[{time.strftime('%H:%M:%S')}] Action Execution: add_to_cart(id={shoe_id}) -> '{final_name}' at ${final_price}, sourced directly from database")
        return f"Success: Added {final_name} to cart for ${final_price}."

    @observe(as_type="span", name="Tool_Execution")
    def remove_from_cart(item_id: str) -> str:
        """Remove one specific item from the cart, identified by its exact id from CURRENT CART."""
        cart_removals.append(item_id)
        trace.append(f"[{time.strftime('%H:%M:%S')}] Action Execution: remove_from_cart('{item_id}')")
        return f"Success: Removed item {item_id} from cart."

    @observe(as_type="span", name="Tool_Execution")
    def clear_cart() -> str:
        """Remove every item from the cart in one action."""
        cart_cleared.append(True)
        trace.append(f"[{time.strftime('%H:%M:%S')}] Action Execution: clear_cart()")
        return "Success: Cart cleared."

    return add_to_cart, remove_from_cart, clear_cart


# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
@observe(name="Veloxa_Agent_Flow")
def run_agent(
    safe_text: str,
    history: list,
    current_cart: list,
    trace: list,
    cart_actions: list,
    cart_removals: list,
    cart_cleared: list,
    image_part: types.Part | None = None,
) -> dict:
    search_query = build_search_query(safe_text, history)
    relevant_shoes = retrieve_relevant_shoes(search_query, trace)

    # Value-based routing: a premium item in the mix earns the stronger model
    is_premium = any(shoe.get("financial_tier") == "Premium" for shoe in relevant_shoes)
    model_name = "gemini-3.5-flash" if is_premium else "gemini-2.5-flash"
    get_client().update_current_span(metadata={"model_tier": "premium" if is_premium else "commodity"})
    trace.append(
        f"[{time.strftime('%H:%M:%S')}] Router: "
        f"{'Premium item detected' if is_premium else 'Commodity-only query'} - routing to {model_name}."
    )

    history_str = "\n".join([f"{msg['role'].upper()}: {msg['text']}" for msg in history[-3:]])
    system_instruction = f"""
    You are the VELOXA AI Concierge - an enterprise omnichannel shopping assistant.
    RETRIEVED INVENTORY: {json.dumps(relevant_shoes)}
    CURRENT CART: {json.dumps(current_cart)}
    STORE POLICIES: {json.dumps(store_policies)}

    DIRECTIVES:
    1. If the user provides an image, use Visual Search to find the closest match in RETRIEVED INVENTORY.
    2. If the user's message is a short follow-up (e.g. "add it", "yes", "that one") referring to a shoe already discussed in HISTORY, use the exact shoe from HISTORY - never ask them to repeat information they already gave you.
    3. Only recommend items from RETRIEVED INVENTORY for new product suggestions. If nothing there fits, say so honestly.
    4. If the user asks to buy or add an item to their cart, call `add_to_cart` with that shoe's numeric "id" field from RETRIEVED INVENTORY - never pass a name or price, only the id. Only say an item was added if you actually called the tool this turn.
    5. If the user asks to remove one specific item, find the best-matching item in CURRENT CART by name and call `remove_from_cart` with that exact item's "id" from CURRENT CART - never invent an id.
    6. If the user asks to remove several items, call `remove_from_cart` once per item.
    7. If the user asks to clear, empty, or remove everything, call `clear_cart` instead of calling remove_from_cart repeatedly.
    8. If CURRENT CART is empty and the user asks to remove something, tell them honestly rather than calling a tool.
    9. You must ONLY output strictly formatted JSON matching this exact structure:
    {{
        "reply": "Your conversational reply...",
        "recommendations": [{{"id": 1, "match_percentage": 95, "reason": "Why it fits.", "recommended_color": "Red"}}]
    }}
    Do NOT wrap the response in markdown code blocks. Output raw JSON.
    """

    add_to_cart_tool, remove_from_cart_tool, clear_cart_tool = make_cart_tools(
        trace, cart_actions, cart_removals, cart_cleared
    )

    # Gemini 3.x performs best with default sampling - Google's own guidance advises
    # against overriding temperature for this generation, unlike 2.5.
    if model_name == "gemini-3.5-flash":
        agent_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[add_to_cart_tool, remove_from_cart_tool, clear_cart_tool],
        )
    else:
        agent_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            tools=[add_to_cart_tool, remove_from_cart_tool, clear_cart_tool],
        )

    user_parts = []
    if image_part:
        trace.append(f"[{time.strftime('%H:%M:%S')}] Vision: Processing multimodal image input...")
        user_parts.append(image_part)
    user_parts.append(types.Part.from_text(text=f"HISTORY:\n{history_str}\nUSER: {safe_text}"))
    contents = [types.Content(role="user", parts=user_parts)]

    try:
        trace.append(f"[{time.strftime('%H:%M:%S')}] Orchestrator: Calling {model_name}...")
        response = client.models.generate_content(
            model=model_name, contents=contents, config=agent_config
        )

        if response.function_calls:
            trace.append(f"[{time.strftime('%H:%M:%S')}] Agent: Tool execution requested.")
            contents.append(response.candidates[0].content)

            tool_map = {
                "add_to_cart": add_to_cart_tool,
                "remove_from_cart": remove_from_cart_tool,
                "clear_cart": clear_cart_tool,
            }

            tool_responses = []
            for call in response.function_calls:
                fn = tool_map.get(call.name)
                if fn:
                    result = fn(**call.args)
                    tool_responses.append(
                        types.Part.from_function_response(name=call.name, response={"result": result})
                    )
            contents.append(types.Content(role="user", parts=tool_responses))

            trace.append(f"[{time.strftime('%H:%M:%S')}] Orchestrator: Returning tool output for final synthesis...")
            response = client.models.generate_content(
                model=model_name, contents=contents, config=agent_config
            )

        raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_text)
        trace.append(f"[{time.strftime('%H:%M:%S')}] Orchestrator: Successfully parsed JSON response.")
        return data

    except json.JSONDecodeError:
        trace.append(f"[{time.strftime('%H:%M:%S')}] Error: Failed to parse JSON from LLM.")
        return {"reply": "I encountered an error structuring my response.", "recommendations": []}

    except Exception as e:
        trace.append(f"[{time.strftime('%H:%M:%S')}] Error: Gemini request failed - {type(e).__name__}: {e}")
        return {
            "reply": "I'm experiencing high demand right now and couldn't process that. Please try again in a moment.",
            "recommendations": [],
        }


# ==========================================
# API
# ==========================================
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    cart: list[dict] = []
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_base64: str | None = None
    image_mime_type: str | None = None


@app.get("/")
def read_root():
    return {"message": "Veloxa backend is running"}


@app.get("/admin/metrics")
def admin_metrics():
    langfuse = get_client()
    to_ts = datetime.now(timezone.utc)
    from_ts = to_ts - timedelta(days=7)

    result = {
        "window_days": 7,
        "total_conversations": None,
        "avg_response_seconds": None,
        "escalations": None,
        "tool_calls": None,
        "error": None,
    }

    try:
        query = json.dumps({
            "view": "observations",
            "metrics": [
                {"measure": "count", "aggregation": "count"},
                {"measure": "latency", "aggregation": "avg"},
            ],
            "dimensions": [{"field": "name"}],
            "filters": [],
            "fromTimestamp": from_ts.isoformat(),
            "toTimestamp": to_ts.isoformat(),
        })
        raw = langfuse.api.metrics.metrics(query=query)
        data = raw.model_dump()["data"] if hasattr(raw, "model_dump") else raw["data"]
        by_name = {row.get("name"): row for row in data}

        if "Chat_Request" in by_name:
            row = by_name["Chat_Request"]
            result["total_conversations"] = int(row.get("count_count", 0))
            result["avg_response_seconds"] = round(float(row.get("avg_latency", 0)) / 1000, 1)

        if "Tool_Execution" in by_name:
            result["tool_calls"] = int(by_name["Tool_Execution"].get("count_count", 0))

    except Exception as e:
        result["error"] = f"overview query failed: {type(e).__name__}: {e}"

    try:
        esc_query = json.dumps({
            "view": "observations",
            "metrics": [{"measure": "count", "aggregation": "count"}],
            "dimensions": [],
            "filters": [
                {"column": "metadata", "operator": "contains", "key": "escalated", "value": "true", "type": "stringObject"}
            ],
            "fromTimestamp": from_ts.isoformat(),
            "toTimestamp": to_ts.isoformat(),
        })
        esc_raw = langfuse.api.metrics.metrics(query=esc_query)
        esc_data = esc_raw.model_dump()["data"] if hasattr(esc_raw, "model_dump") else esc_raw["data"]
        result["escalations"] = int(esc_data[0]["count_count"]) if esc_data else 0
    except Exception as e:
        if not result["error"]:
            result["error"] = f"escalation query failed: {type(e).__name__}: {e}"

    return result


@app.post("/chat")
@observe(name="Chat_Request")
def chat(request: ChatRequest):
    trace: list[str] = [f"[{time.strftime('%H:%M:%S')}] System: Request received"]
    cart_actions: list[dict] = []
    cart_removals: list[str] = []
    cart_cleared: list[bool] = []

    with propagate_attributes(
        user_id="enterprise-shopper",
        session_id=request.session_id,
        tags=["production", "fastapi-backend"],
    ):
        safe_text = scrub_pii(request.message, trace)

        if check_hitl_escalation(safe_text, trace):
            get_client().update_current_span(metadata={"escalated": "true"})
            get_client().flush()
            return {
                "reply": "I am escalating your request to a specialized human agent.",
                "recommendations": [],
                "trace_log": trace,
                "cart_actions": cart_actions,
                "cart_removals": cart_removals,
                "cart_cleared": False,
                "escalate": True,
            }

        image_part = None
        if request.image_base64:
            image_bytes = base64.b64decode(request.image_base64)
            image_part = types.Part.from_bytes(
                data=image_bytes, mime_type=request.image_mime_type or "image/jpeg"
            )

        result = run_agent(
            safe_text, request.history, request.cart, trace,
            cart_actions, cart_removals, cart_cleared, image_part,
        )
        get_client().flush()

        return {
            "reply": result.get("reply", "Error communicating with the Concierge."),
            "recommendations": result.get("recommendations", []),
            "trace_log": trace,
            "cart_actions": cart_actions,
            "cart_removals": cart_removals,
            "cart_cleared": len(cart_cleared) > 0,
            "escalate": False,
        }