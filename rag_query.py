import os
from typing import Optional

from crewai.rag.core.base_embeddings_callable import normalize_embeddings
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import dotenv
from sympy.physics.units import temperature

dotenv.load_dotenv()

#--------CONFIG---------
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"
TOP_K_VECTOR = 5
TOP_K_GRAPH = 3
TOP_K_FINAL = 5
#-----------------------

#---------PROMPTS--------

RECOMMENDATION_PROMPT = PromptTemplate(
    input_variables=["query", "products"],
    template="""
You are an expert e-commerce product recommendation assistant.

A user is looking for: "{query}"

Based on the following products retrieved from a knowledge graph,
provide clear and helpful recommendations. For each product you recommend:
- Explain specifically WHY it matches the user's needs
- Highlight key features that make it a good fit
- Mention the rating and price if available
- Be concise but informative

Retrieved products:
{products}

Give your top recommendations with explanations. Be conversational and helpful.
If none of the products are a strong match, say so honestly.
"""
)

SIMILARITY_PROMPT = PromptTemplate(
    input_variables=["source_product", "similar_products"],
    template="""
You are an expert e-commerce assistant helping users discover related products.

The user is viewing:
{source_product}

These similar products were found using the knowledge graph
(shared features, categories, and co-purchase patterns):
{similar_products}

Explain why each similar product is related to the original. Focus on:
- Shared features or use cases
- Why someone who likes the original might enjoy each suggestion
- Any meaningful differences worth noting

Be helpful and conversational.
"""
)
#----------------------------------------------

#-----------NEO4J CONNECTOR--------------------
class Neo4jConnector:
    def __init__(self, url: str, user: str, password: str):
        self.driver = GraphDatabase.driver(url, auth=(user, password))
        self.driver.verify_connectivity()
        print("Connected to Neo4j Aura")

    def close(self):
        self.driver.close()

    def query(self, cypher:str, params={}) ->list[dict]:
        """Run a read query and return all rows as dicts"""
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return [r.data() for r in result]

#------------GRAPH RETRIEVER----------------------
class GraphRetriever:
    def __init__(self, neo4j: Neo4jConnector, embed_model: SentenceTransformer):
        self.neo4j = neo4j
        self.embed_model = embed_model

    def semantic_search(self, query: str, top_k: int = TOP_K_GRAPH) -> list[dict]:
        """Find products via vector similarity on the Neo4j vector index"""
        embedding = self.embed_model.encode(query, normalize_embeddings=True).tolist()

        return self.neo4j.query(
            """
            CALL db.index.vector.queryNodes(
                'product_embedding_index', $top_k, $embedding
            )
            YIELD node AS p, score
            OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
            OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            RETURN
                p.asin           AS asin,
                p.title          AS title,
                p.price          AS price,
                p.average_rating AS rating,
                p.rating_number  AS num_ratings,
                b.name           AS brand,
                collect(DISTINCT c.name)[0..3] AS categories,
                collect(DISTINCT f.name)[0..6] AS features,
                score
            ORDER BY score DESC
            """,
            {"embedding": embedding, "top_k": top_k},
        )
    #---- GRAPH TRAVERSAL------------------------------------

    def similar_graph(self, asin:str, top_k: int = TOP_K_GRAPH) -> dict:
        """
                Find similar products using 3 graph signals:
                  - Shared features       (weight: 3)
                  - Same category         (weight: 2)
                  - Co-purchased together (weight: 1)
                """
        source = self.neo4j.query(
            """
            MATCH (p:Product {asin: $asin})
            OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
            OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            RETURN
                p.asin AS asin, p.title AS title,
                p.price AS price, p.average_rating AS rating,
                b.name AS brand,
                collect(DISTINCT c.name) AS categories,
                collect(DISTINCT f.name)[0..8] AS features
            """,
            {"asin": asin},
        )
        if not source:
            return {"source": None, "similar": []}
        similar = self.neo4j.query(
            """
            MATCH (src:Product {asin: $asin})

            // shared features
            OPTIONAL MATCH (src)-[:HAS_FEATURE]->(f:Feature)<-[:HAS_FEATURE]-(pf:Product)
            WHERE pf.asin <> $asin
            WITH src, collect({p: pf, w: 3 * 1}) AS feat_hits

            // same category
            OPTIONAL MATCH (src)-[:BELONGS_TO]->(c:Category)<-[:BELONGS_TO]-(pc:Product)
            WHERE pc.asin <> $asin
            WITH src, feat_hits, collect({p: pc, w: 2}) AS cat_hits

            // co-purchased
            OPTIONAL MATCH (src)-[:CO_PURCHASED]-(pco:Product)
            WHERE pco.asin <> $asin
            WITH feat_hits + cat_hits + collect({p: pco, w: 1}) AS all_hits

            UNWIND all_hits AS hit
            WITH hit.p AS p, sum(hit.w) AS graph_score
            WHERE p IS NOT NULL

            OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
            OPTIONAL MATCH (p)-[:BELONGS_TO]->(cat:Category)
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            RETURN
                p.asin           AS asin,
                p.title          AS title,
                p.price          AS price,
                p.average_rating AS rating,
                b.name           AS brand,
                collect(DISTINCT cat.name)[0..3] AS categories,
                collect(DISTINCT f.name)[0..6]   AS features,
                graph_score
            ORDER BY graph_score DESC
            LIMIT $top_k
            """,
            {"asin": asin, "top_k": top_k},
        )

        return {"source": source[0], "similar": similar}

    #------HYBRID = VECTOR + GRAPH ENRICHMENT -----------------------

    def hybrid_search(self, query: str, top_k: int = TOP_K_GRAPH) -> list[dict]:
        """
               Vector search → graph enrichment → re-rank.
               Graph adds brand, features, co-purchased context to each candidate.
               Re-rank adds a small rating bonus on top of vector score.
               """
        vector_results = self.semantic_search(query, top_k=TOP_K_VECTOR)
        if not vector_results:
            return []

        asins = [r["asin"] for r in vector_results]
        score_map = {r["asin"]: r["score"] for r in vector_results}
        enriched = self.neo4j.query(
            """
            UNWIND $asins AS asin
            MATCH (p:Product {asin: asin})
            OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
            OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            OPTIONAL MATCH (p)-[:CO_PURCHASED]-(cp:Product)
            RETURN
                p.asin           AS asin,
                p.title          AS title,
                p.price          AS price,
                p.average_rating AS rating,
                p.rating_number  AS num_ratings,
                b.name           AS brand,
                collect(DISTINCT c.name)[0..4]   AS categories,
                collect(DISTINCT f.name)[0..8]   AS features,
                collect(DISTINCT cp.title)[0..3] AS also_bought
            """,
            {"asins": asins},
        )
        # Re-rank: vector score + small rating bonus
        for row in enriched:
            vscore = score_map.get(row["asin"], 0)
            rating_bonus = (float(row["rating"] or 0) / 5) * 0.1
            row["final_score"] = round(vscore + rating_bonus, 4)

        enriched.sort(key=lambda x: x["final_score"], reverse=True)
        return enriched[:top_k]

#-------FORMATTERS-----------------------------------

def format_products(products:list[dict]) -> str:
    lines = []
    for i, p in enumerate(products, 1):
        price = f"${float(p['price']):.2f}" if p.get("price") else "N/A"
        features = ", ".join(p.get("features") or []) or "N/A"
        categories = ", ".join(p.get("categories") or []) or "N/A"
        also_bought = p.get("also_bought") or []

        block = (
            f"Product {i}: {p.get('title', 'N/A')}\n"
            f"  Brand      : {p.get('brand') or 'N/A'}\n"
            f"  Price      : {price}\n"
            f"  Rating     : {p.get('rating') or 'N/A'}/5 ({p.get('num_ratings') or '?'} reviews)\n"
            f"  Category   : {categories}\n"
            f"  Features   : {features}"

        )
        if also_bought:
            block += f"\n  Often bought with: {', '.join(str(x) for x in also_bought)}"
        lines.append(block)
    return "\n\n".join(lines)

def format_source(p:dict) -> str:
    price = f"${float(p['price']):.2f}" if p.get("price") else "N/A"
    features = ", ".join(p.get("features") or []) or "N/A"
    return (
        f"{p.get('title', 'N/A')}\n"
        f"  Brand    : {p.get('brand') or 'N/A'}\n"
        f"  Price    : {price}\n"
        f"  Rating   : {p.get('rating') or 'N/A'}/5\n"
        f"  Features : {features}"
    )

#------------MAIN RAG CLASS-----------------

class EcommerceRAG:
    def __init__(
            self,
            neo4j_url:str = NEO4J_URL,
            neo4j_user: str = NEO4J_USER,
            neo4j_password: str = NEO4J_PASSWORD,
            gemini_api_key: str = GEMINI_API_KEY,
    ):
        print("Initializing EcommerceRAG.....")

        self.neo4j = Neo4jConnector(neo4j_url, neo4j_user, neo4j_password)

        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        self.retriever = GraphRetriever(self.neo4j, self.embed_model)

        print(f"Connecting to Gemini: {GEMINI_MODEL}")
        self.llm = ChatGoogleGenerativeAI(
            model = GEMINI_MODEL,
            google_api_key = gemini_api_key,
            temperature = 0.4,
            max_output_tokens = 1024,
        )

        self._recommend_chain = (
            RunnablePassthrough()
            | RECOMMENDATION_PROMPT
            | self.llm
            | StrOutputParser()

        )

        self._similar_chain = (
            RunnablePassthrough()
            | SIMILARITY_PROMPT
            | self.llm
            | StrOutputParser()

        )

        print("EcommerceRAG ready!\n")

    def close(self):
        self.neo4j.close()

    def semantic_search(

            self,
            query: str,
            top_k: int = TOP_K_FINAL,
            explain: bool = False,
    )-> dict:
        """
        Hybrid vector + graph search.Optionally explain with Gemini.
        Returns:
            {"query":str, "products":list[dict], "explanation":str|None}

        """
        print("Semantic searching...")
        products = self.retriever.hybrid_search(query, top_k=top_k)

        explanation = None
        if explain and products:
            print("Generating explanation...")
            explanation = self._recommend_chain.invoke({
                "query": query,
                "products": format_products(products),
            })

        self._print_search_results(products, explanation)
        return {"query": query, "products": products, "explanation": explanation}

    def similar_products(
            self,
            asin: str,
            top_k: int = TOP_K_GRAPH,
            explain: bool = False,
    ) -> dict:
        """
        Graph traversal to find similar products by shared features,category, and co-purchase signals.
        Optionally explain with Gemini.

        """
        print(f"Finding similar products by shared features: {asin}")
        data = self.retriever.similar_graph(asin, top_k=top_k)

        if not data["source"]:
            print('ASIN NOT FOUND IN NEO4J')
            return data

        explanation = None
        if explain and data["similar"]:
            print("Generating explanation...")
            explanation = self._similar_chain.invoke({
                "source_product": format_source(data["source"]),
                "similar_products": format_products(data["similar"]),
            })

        src = data["source"]
        print(f"\n Source: {src.get('title', 'N/A')[:70]}")
        count = len(data['similar'])
        print(f"\n Similar : {count} products found")
        #print(f" Similar: {len(data["similar"])} products found")
        for i, p in enumerate(src["similar_products"], 1):
            score = p.get("graph_score", 0)
            print(f"{i}. [{score}pts] {str(p.get('title',''))[:65]}")
        if explanation:
            print(f"\n📖 Explanation:\n{explanation}")
        return {**data, "explanation": explanation}

    def recommend_and_explain(self, query: str, top_k: int = TOP_K_FINAL) -> dict:
        """
        Main conversational entry point.
        Runs the full pipeline: vector search → graph enrichment → Gemini.
        """
        return self.semantic_search(query, top_k=top_k, explain=True)

    def _print_search_results(self, products, explanation):
        if not products:
            print('No products found')
            return
        print(f"Top {len(products)} products:")
        print("-" * 65)
        for i, p in enumerate(products, 1):
            title = str(p.get("title") or "")[:65]
            price = f"${float(p['price']):.2f}" if p.get("price") else "N/A"
            rating = p.get("rating") or "N/A"
            score = p.get("final_score") or p.get("score") or 0
            print(f" {i}. {title}")
            print(f"     💰 {price}  ⭐ {rating}/5  🎯 {score:.3f}")
        if explanation:
            print("\n" + "-" * 65)
            print(explanation)
            print("-" * 65)

if __name__ == "__main__":
    import sys
    import time
    if "<your-instance>" in NEO4J_URL or "<your-gemini-api-key>" in GEMINI_API_KEY:
        print(
            "❌  Fill in your credentials at the top of this file or\n"
            "    pass them as arguments to EcommerceRAG().\n"
            "    Gemini API key → https://aistudio.google.com/app/apikey\n"
        )
        sys.exit(1)

    rag = EcommerceRAG()
    try:
        #rag.semantic_search("wireless earbuds long battery life",explain=False)

        #time.sleep(15)

        rag.recommend_and_explain("best laptop for video editing under $1000")

    finally:
        rag.close()











