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
 


