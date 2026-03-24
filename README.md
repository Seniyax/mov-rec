# E-Commerce Recommendation Engine
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-Aura-008CC1?style=flat&logo=neo4j&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=flat)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-4285F4?style=flat&logo=google&logoColor=white)
## Overview
This project builds an intelligent **e-commerce product recommendation engine** that combines **Knowledge Graphs** and **Retrieval-Augmented Generation (RAG)** to deliver explainable, context-aware product recommendations.
 
Unlike traditional recommendation systems that rely purely on collaborative filtering or basic vector similarity, this system enriches recommendations with structured graph relationships — understanding not just *what* users bought, but *why* products are related.

**Dataset** - Amazon Electronics Reviews (McAuley-Lab/Amazon-Reviews-2023) — 50,000 reviews, 20,000 products.
## Architecture
```
                  ┌─────────────────────────────────────────────────────────────┐
                  │                        User / UI                            │
                  │                    (Streamlit Frontend)                     │
                  └──────────────────────────┬──────────────────────────────────┘
                                             │ HTTP
                  ┌──────────────────────────▼──────────────────────────────────┐
                  │                      FastAPI Backend                        │
                  │         /recommend   /trending   /explain   /categories     │
                  └────────────┬─────────────────────────────┬──────────────────┘
                               │                             │
                  ┌────────────▼──────────┐    ┌─────────────▼────────────────┐
                  │   Recommendation      │    │       RAG Query Layer         │
                  │       Engine          │    │  sentence-transformers +      │
                  │  - Collaborative      │    │  Neo4j Vector Index +         │
                  │  - Trending           │    │  LangChain + Gemini 2.0 Flash │
                  │  - Blended            │    └─────────────────────────────--┘
                  └────────────┬──────────┘                 │
                               │                            │
                  ┌────────────▼────────────────────────────▼────────────────────┐
                  │                     Neo4j Aura (Cloud)                       │
                  │                                                              │
                  │  Nodes: Product · Brand · Category · Feature · User          │
                  │  Edges: MADE_BY · BELONGS_TO · HAS_FEATURE · REVIEWED        │
                  │         CO_PURCHASED                                         │
                  │  Vector Index: product_embedding_index (384 dims, cosine)    │
                  └──────────────────────────────────────────────────────────────┘
```
 
## Features
| Feature | Description |
|---|---|
| **Collaborative Filtering** | Finds products reviewed by the same users |
| **Trending Products** | Ranks by `avg_rating × log(review_count)` per category |
| **Semantic Search** | Vector similarity search via Neo4j vector index |
| **AI Explanations** | Gemini 2.0 Flash explains *why* products are recommended |
| **Graph Traversal** | Multi-hop relationship discovery through the knowledge graph |

## Project Structure
```
RAG/
├──Data
|     └── extract_data.ipynb         Load dataset from hugging face and extract entities and build KG
|     └── semantic_searcher.ipynb    Upload KG to neo4j aura and generate vector embeddings
├──rag_query.py                      RAG pipeline
├──recommendation_engine.py          Recommendation Logic
├──api.py                            FastAPI REST backend
├──ui.py                             Streamlit frontend
└──README.md
````
## Knowledge Graph Schema
### Nodes
| Label | Key Properties |
|---|---|
| `Product` | `asin`, `title`, `price`, `average_rating`, `rating_number` |
| `Brand` | `name` |
| `Category` | `name` |
| `Feature` | `name` |
| `User` | `user_id` |
### Relationships
| Relationship | From → To | Description |
|---|---|---|
| `MADE_BY` | Product → Brand | Product manufacturer |
| `BELONGS_TO` | Product → Category | Product category path |
| `HAS_FEATURE` | Product → Feature | Extracted product features |
| `REVIEWED` | User → Product | User review (rating, sentiment, date) |
| `CO_PURCHASED` | Product ↔ Product | Shared reviewers across both products |
## ⚙️ Setup & Installation
### Requirements
- Python 3.11+
- [Neo4j Aura](https://console.neo4j.io) free account
- [Google AI Studio](https://aistudio.google.com/apikey) API key (free)
### 1. Clone the repository
```bash
git clone https://github.com/Seniyax/mov-rec.git
cd mov-rec
```
### 2. Install dependencies
```bash
pip install datasets==2.17.0 pandas pyarrow tqdm
pip install spacy networkx
python -m spacy download en_core_web_sm
 
pip install neo4j sentence-transformers torch
pip install langchain langchain-google-genai google-generativeai
pip install fastapi uvicorn streamlit requests
```
### API & UI
Start the FastAPI backend:
```bash
uvicorn api:app --reload --port 8000
```
In a separate terminal, start the Streamlit UI:
```bash
streamlit run ui.py
```
#### API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/recommend/{asin}` | Blended recommendations for a product |
| `GET` | `/trending/{category}` | Trending products in a category |
| `GET` | `/explain/{asin}` | AI-generated explanation for recommendations |
| `GET` | `/categories` | List all categories (for UI dropdown) |

## 🛠️ Tech Stack
 
| Component | Technology |
|---|---|
| Graph Database | Neo4j Aura (cloud) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector Search | Neo4j built-in vector index |
| LLM | Google Gemini 2.0 Flash |
| RAG Framework | LangChain |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data Processing | pandas, spaCy, NetworkX |
| Dataset | Amazon Reviews 2023 (McAuley-Lab) |

# Recommendation Strategies
 
### Collaborative Filtering
 
Finds products that were reviewed by the same users as the input product.
 
```
Score = (shared_reviewers × 0.6) + (avg_rating × 0.4)
```
 
### Trending
 
Ranks products within a category by balancing rating quality with review volume.
 
```
Score = avg_rating × log(review_count + 1)
```
 
The logarithm prevents a product with 10,000 mediocre reviews from outranking one with 500 excellent reviews.

## 📚 References
 
- [Amazon Reviews 2023 Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- [Neo4j Vector Search Documentation](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [LangChain Documentation](https://python.langchain.com)
- [Google Gemini API](https://ai.google.dev)
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag)
 
## DEMO VIDEO

## Author
 
Built as a student project exploring the intersection of **Knowledge Graphs** and **Retrieval-Augmented Generation** for intelligent e-commerce recommendations.
 
---

## 📄 License
 
This project is for educational purposes. The Amazon Reviews dataset is subject to its own [terms of use](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
