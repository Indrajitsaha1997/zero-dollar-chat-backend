from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ingest_documents, get_rag_chain, store
import shutil, os

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = None

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.get("/")
def root():
    return {"status": "RAG Chatbot running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("docs", exist_ok=True)
    dest = f"docs/{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    count = ingest_documents(dest)
    return {"message": f"Ingested {count} chunks from {file.filename}"}

@app.post("/chat")
async def chat(req: ChatRequest):
    global rag_chain
    if rag_chain is None:
        rag_chain = get_rag_chain()

    result = rag_chain.invoke(
        {"input": req.question},
        config={"configurable": {"session_id": req.session_id}},
    )

    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in result.get("context", [])
    })

    return {
        "answer": result["answer"],
        "sources": sources,
    }

@app.delete("/reset")
def reset_memory():
    global rag_chain
    rag_chain = None
    store.clear()
    return {"message": "Conversation memory cleared"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)