from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import os, uuid

from app.logging_mw import LoggingMiddleware
from app.rag import pdf_to_text, chunk, db

app = FastAPI(title="LLM RAG Runner")
app.add_middleware(LoggingMiddleware)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    up_dir = "app/storage/uploads"
    os.makedirs(up_dir, exist_ok=True)
    fid = f"{uuid.uuid4()}_{file.filename}"
    fpath = os.path.join(up_dir, fid)
    with open(fpath, "wb") as f:
        f.write(await file.read())

    text = pdf_to_text(fpath)
    chunks = chunk(text)
    db.add_texts(chunks)
    return {"uploaded": file.filename, "chunks": len(chunks)}

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

def call_llm(prompt: str) -> str:
    # TODO: 실제 LLM API로 교체. 지금은 흐름 확인용 더미 응답.
    return "(MOCK ANSWER) " + (prompt[:180] + ("..." if len(prompt) > 180 else ""))

@app.post("/ask")
def ask(payload: AskRequest):
    hits = db.search(payload.question, k=payload.top_k)
    if not hits:
        return {"answer": "문서에서 관련 근거를 찾지 못했습니다.", "evidence": []}
    evidence_texts = [h[2] for h in hits]
    context = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(evidence_texts))
    prompt = f"""다음 근거를 바탕으로 질문에 답변하세요.
근거:
{context}

질문: {payload.question}
규칙: 근거가 부족하면 '근거 부족'이라 답하고 가능한 경우 [번호]로 근거를 표시하세요."""
    answer = call_llm(prompt)
    return {
        "answer": answer,
        "evidence": [{"rank": i+1, "snippet": t} for i, t in enumerate(evidence_texts)]
    }
