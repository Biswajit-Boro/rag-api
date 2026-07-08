from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()

client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)


class DocumentSubmission(BaseModel):
    user_name: str
    content: str


@app.post("/documents")
def add_document(submission: DocumentSubmission):
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]

    if not chunks:
        return {"error": "No content provided"}

    collection.add(
        ids=[f"{submission.user_name}-chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": "profile", "user_name": submission.user_name, "chunk_index": i}
            for i in range(len(chunks))
        ],
    )
    return {
        "message": f"Added {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
        "preview": chunks[0],  # first chunk so the user can verify the split looks right
    }


@app.get("/users")  # NEW: list all distinct users currently stored
def list_users():
    all_data = collection.get(include=["metadatas"])
    user_names = sorted(set(m["user_name"] for m in all_data["metadatas"] if "user_name" in m))
    return {"users": user_names, "count": len(user_names)}


@app.delete("/documents/{user_name}")  # NEW: delete a user's entire profile
def delete_document(user_name: str):
    existing = collection.get(where={"user_name": user_name})

    if not existing["ids"]:
        return {"error": f"No profile found for user '{user_name}'"}

    collection.delete(where={"user_name": user_name})
    return {
        "message": f"Deleted profile for user '{user_name}'",
        "chunks_removed": len(existing["ids"]),
    }


@app.get("/ask")
def ask(question: str, user: str = None):
    query_params = {
        "query_texts": [question],
        "n_results": 2,
    }
    if user:
        query_params["where"] = {"user_name": user}

    results = collection.query(**query_params)

    if not results["documents"][0]:
        return {
            "question": question,
            "answer": "No relevant profile found.",
            "context_used": [],
            "filtered_by_user": user,
        }

    context = "\n\n".join(results["documents"][0])

    augmented_prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}"""

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
        "filtered_by_user": user,
    }


# Serve the simple frontend at the root URL
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")