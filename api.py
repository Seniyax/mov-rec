import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.app.wordnet_app import explanation
from pydantic import BaseModel
from typing import Optional
import dotenv
import os

from recommendation_engine import RecommendationEngine
from rag_query import EcommerceRAG, GEMINI_API_KEY

dotenv.load_dotenv()

#-----CONFIG--------------------------------
NEO4J_URI = os.getenv("NEO4J_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#--------------------------------------------

#-----APP SETUP-----------------------------
app = FastAPI(
    title="Ecommerce RAG",
    description="Knowledge Graph + RAG powered product recommendations",
    version="1.0.0",
     )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: RecommendationEngine = None
rag: EcommerceRAG = None

@app.on_event("startup")
def startup():
    global engine,rag
    print("🚀 Starting up — connecting to Neo4j and loading models...")
    engine = RecommendationEngine(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
    )
    rag = EcommerceRAG(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        GEMINI_API_KEY,
    )
    print("API ready")

@app.on_event("shutdown")
def shutdown():
    if engine: engine.close()
    if rag: rag.close()


class Product(BaseModel):
    asin: Optional[str] = None
    title: Optional[str] = None
    brand: Optional[str] = None
    avg_rating: Optional[float] = None
    price: Optional[float] = None
    review_count: Optional[int] = None
    shared_reviewers: Optional[int] = None
    trending_score: Optional[float] = None
    blended_score: Optional[float] = None
    source: Optional[str] = None

    model_config = {"json_encoders": {float: lambda v: None if v != v else v}}


class RecommendResponse(BaseModel):
    asin: str
    results: list[Product]


class TrendingResponse(BaseModel):
    category: str
    results: list[Product]


class ExplainResponse(BaseModel):
    asin: str
    explanation: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend/{asin}",response_model=RecommendResponse)
def recommend(asin:str,top_k:int=8):
    try:
        df = engine.blended(asin=asin,top_k=top_k)
        if df.empty:
            return {"asin":asin,"results":[]}
        df = df.where(df.notna(),other=None)
        results = df.to_dict(orient="records")
        return {"asin":asin,"results":results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending/{category}",response_model=TrendingResponse)
def trending(category: str,top_k:int=8):
    try:
        import math


        df = engine.trending(category=category,top_k=top_k)
        if df.empty:
            return {"asin":category,"results":[]}
        df = df.where(df.notna(),other=None)
        results = df.to_dict(orient="records")
        return {"category":category,"results":results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{asin}",response_model=ExplainResponse)
def explain(asin:str):
    try:
        result = rag.recommend_and_explain(query=asin)
        if isinstance(result,dict):
            explanation = result.get("explanation") or ""
        else:
            explanation = str(result) if result else ""
        explanation = explanation.replace("$nan", "price not available")
        explanation = explanation.replace('"nan"', "price not available")

        if not explanation:
            explanation = "No explanation could be generated. The product may not have enough graph data."

        return {"asin":asin,"explanation":explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
def categories(keyword: str = ""):
    """
    List categories — used to populate the dropdown in the UI.
    Pass an optional keyword to filter results.
    """
    try:
        df = engine.find_categories(keyword=keyword)
        cats = df["category"].tolist() if not df.empty and "category" in df.columns else []
        return {"categories": cats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
