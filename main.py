from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import os
import time
import pdfplumber
from astrapy import DataAPIClient

# Load environment variables
load_dotenv()

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
COLLECTION_NAME = "insurance_dataset"

# Astra DB setup
# ✅ Async DB access
def get_async_db():
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    client = DataAPIClient()
    return client.get_async_database(endpoint, token=token)


@app.post("/connect")
async def connect():
    try:
        db = get_db()
        collections = await db.list_collection_names()
        return {"success": True, "databaseId": db.id, "collections": collections}
    except Exception as e:
        return {"success": False, "error": str(e)}



@app.post("/collection")
async def create_collection(request: Request):
    body = await request.json()
    collection = body.get("collection")
    try:
        db = get_db()
        existing = await db.list_collection_names()
        if collection in existing:
            await db.drop_collection(collection)
        await db.create_collection(collection, {
            "vector": {"dimension": 1536, "metric": "cosine"},
            "service": {"provider": "datastax", "model": "NV-Embed-QA"},
            "fieldToEmbed": "$vectorize"
        })
        return {"success": True, "message": "Collection created/reset."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with pdfplumber.open(file.file) as pdf:
            text = ''.join([page.extract_text() or "" for page in pdf.pages])
        cleaned_text = " ".join(text.split())

        chunks = []
        CHUNK_SIZE = 500
        for i in range(0, len(cleaned_text), CHUNK_SIZE):
            chunk = cleaned_text[i:i + CHUNK_SIZE]
            chunks.append({
                "_id": f"chunk-{int(time.time())}-{i}",
                "$vectorize": chunk,
                "type": "pdf",
                "source": file.filename
            })

        db = get_async_db()
        collection = db.get_collection(COLLECTION_NAME)
        result = await collection.insert_many(chunks)

        return {
            "success": True,
            "insertedCount": len(result.inserted_ids),  # ✅ correct fix
            "fileName": file.filename,
            "chunkSize": CHUNK_SIZE
        }
    except Exception as e:
        return {"success": False, "error": str(e)}



@app.post("/insert")
async def insert_data(request: Request):
    try:
        body = await request.json()
        data = body.get("data")
        if not isinstance(data, list):
            return {"error": "'data' must be an array."}

        db = get_async_db()
        collection = await db.get_collection(COLLECTION_NAME)
        result = await collection.insert_many(data)
        return {"success": True, "insertedCount": result["insertedCount"]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/search")
async def search_data(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        limit = body.get("limit", 3)

        if not query:
            return {"success": False, "error": "Missing 'query' in request body"}

        db = get_async_db()
        collection: AsyncCollection = db.get_collection(COLLECTION_NAME)

        # Run hybrid vector search (no extra metadata in response)
        async_cursor = collection.find_and_rerank(
            sort={"$hybrid": query},
            limit=limit,
            projection={"*": True}  # Optional: mimic { "*": 1 } in Node.js
        )

        results = []
        async for result in async_cursor:
            # ✅ Only return the document content (not similarity score etc.)
            results.append(result.document)

        return {"success": True, "results": results}

    except Exception as e:
        return {"success": False, "error": str(e)}




@app.post("/ask")
async def ask_ai(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        limit = body.get("limit", 3)

        if not query:
            return {"success": False, "error": "Missing 'query' in request body"}

        db = get_async_db()
        collection: AsyncCollection = db.get_collection(COLLECTION_NAME)

        # Hybrid search using Datastax auto-vectorization
        async_cursor = collection.find_and_rerank(
            sort={"$hybrid": query},
            limit=limit,
            projection={"*": True}
        )

        # Similarity filtering
        docs = []
        async for r in async_cursor:
            if hasattr(r, "similarity") and r.similarity >= 0.7:
                docs.append(r.document)

        if not docs:
            return {"success": True, "summary": "No matches found."}

        # Create context from matched docs
        context_text = "\n".join(
            [f"Context:\n{doc.get('$vectorize', '')}\n" for doc in docs]
        )

        prompt = f"""Use the following pieces of context to answer the question at the end.

{context_text}

Question: {query}"""

        # Ask OpenAI
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question based ONLY on the provided context."},
                {"role": "user", "content": prompt}
            ]
        )

        summary = completion.choices[0].message.content
        return {"success": True, "summary": summary}

    except Exception as e:
        return {"success": False, "error": str(e)}

