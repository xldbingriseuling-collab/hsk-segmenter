from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jieba
import os

# 加载 HSK 自定义词典
DICT_PATH = os.path.join(os.path.dirname(__file__), "hsk_vocab.txt")
if os.path.exists(DICT_PATH):
    jieba.load_userdict(DICT_PATH)
    print(f"✅ HSK词典加载成功")
else:
    print("⚠️ 未找到HSK词典，使用默认词典")

app = FastAPI(title="HSK Segmenter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SegmentRequest(BaseModel):
    text: str
    
class SegmentResponse(BaseModel):
    tokens: list[str]
    count: int

@app.get("/")
def health():
    return {"status": "ok", "service": "hsk-segmenter"}

@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text不能为空")
    
    tokens = list(jieba.cut(req.text))
    tokens = [t for t in tokens if t.strip()]
    
    return SegmentResponse(tokens=tokens, count=len(tokens))

@app.post("/segment-batch")
def segment_batch(sentences: list[str]):
    """批量分词，一次请求处理整篇文章所有句子"""
    if not sentences:
        raise HTTPException(status_code=400, detail="sentences不能为空")
    
    results = []
    for sentence in sentences:
        tokens = list(jieba.cut(sentence))
        tokens = [t for t in tokens if t.strip()]
        results.append(tokens)
    
    return {"results": results, "sentence_count": len(results)}