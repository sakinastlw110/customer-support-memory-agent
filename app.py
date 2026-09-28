import os
import asyncio
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from hindsight_client import Hindsight
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not HINDSIGHT_API_KEY or not GROQ_API_KEY:
    print("WARNING: HINDSIGHT_API_KEY or GROQ_API_KEY is missing. Check your .env file.")

# Hindsight Cloud client — this is where "memory" is stored and searched
hindsight = Hindsight(
    base_url="https://api.hindsight.vectorize.io",
    api_key=HINDSIGHT_API_KEY,
)

# Groq client — OpenAI-compatible, used to generate the actual reply text
groq = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-120b"  # fast + free-tier friendly on Groq

# One event loop, kept alive for the whole app's lifetime.
# (Hindsight's internal connection gets tied to whichever loop it first
# runs on — asyncio.run() closes its loop after every call, which broke
# things on the second request. Reusing one loop avoids that.)
_loop = asyncio.new_event_loop()


def run_async(coro):
    return _loop.run_until_complete(coro)


def bank_id_for(customer_id: str) -> str:
    """Each customer gets their own isolated memory bank."""
    return f"customer_{customer_id}"


@app.route("/")
def home():
    # Serves the frontend so everything runs on one origin (no CORS needed)
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/seed", methods=["POST"])
def seed():
    """
    Pre-load fake past tickets for a demo customer, so recall has
    something real to find. Call this once before you start chatting.
    Body: { "customer_id": "alice", "tickets": ["...", "..."] }
    """
    data = request.get_json(force=True)
    customer_id = data["customer_id"]
    tickets = data.get("tickets", [])
    bank_id = bank_id_for(customer_id)

    async def _seed_all():
        for ticket in tickets:
            await hindsight.aretain(bank_id=bank_id, content=ticket)

    run_async(_seed_all())

    return jsonify({"status": "seeded", "customer_id": customer_id, "count": len(tickets)})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Body: { "customer_id": "alice", "message": "My login isn't working again" }
    """
    data = request.get_json(force=True)
    customer_id = data["customer_id"]
    message = data["message"]
    bank_id = bank_id_for(customer_id)

    # 1. RECALL — pull anything relevant from this customer's memory
    async def _recall():
        return await hindsight.arecall(bank_id=bank_id, query=message)

    recalled = run_async(_recall())
    memory_lines = [r.text for r in recalled.results]
    memory_context = "\n".join(f"- {line}" for line in memory_lines) or "No past history found for this customer yet."

    # 2. Ask the LLM to reply, using recalled memories as context
    system_prompt = (
        "You are a warm, competent customer support agent. Use the customer's "
        "past history below to personalize your reply — reference specific "
        "past issues naturally, the way a human rep who actually remembers "
        "the customer would. Keep replies short (2-4 sentences).\n\n"
        f"Customer's past history:\n{memory_context}"
    )

    completion = groq.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
    )
    reply = completion.choices[0].message.content

    # 3. RETAIN — store this new interaction so future chats remember it too
    async def _retain():
        await hindsight.aretain(bank_id=bank_id, content=f"Customer said: {message}")

    run_async(_retain())

    return jsonify({"reply": reply, "recalled_memories": memory_lines})


if __name__ == "__main__":
    app.run(debug=False, port=5000)