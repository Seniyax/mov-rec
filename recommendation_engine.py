import os

import pandas as pd
from instructor.cli.batch import results
from neo4j import GraphDatabase
import dotenv
dotenv.load_dotenv()

#-----------CONFIG------------------------
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URI = os.getenv("NEO4J_URL")
#-----------------------------------------


class RecommendationEngine:
    def __init__(self,uri=NEO4J_URI,user=NEO4J_USER,password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        print("Connected to Neo4j")

    def close(self):
        self.driver.close()

    def _query(self,cypher:str,params:dict={}) -> list[dict]:
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher,params)]

    def _show(self,results:list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(results)
        if df.empty:
            print("No results Found")
        else:
            print(df.to_string(index=False))
        return df
    #---------1.COLLABORATIVE FILTERING-------
    def collaborative_filtering(self,asin:str,top_k:int=10) -> pd.DataFrame:
        """
        Item-based collaborative filtering.

        Idea: Find users who reviewed the given product, then see what
        other products those same users also reviewed. Products reviewed
        by many of the same users are likely relevant.

        Ranked by: how many users reviewed both products (shared_reviewers).

        """
        print(f"\nCollaborative filtering -> ASIN: {asin}\n")

        results = self._query("""
            // Find users who reviewed the source product
            MATCH (u:User)-[:REVIEWED]->(source:Product {asin: $asin})
 
            // Find other products those users also reviewed
            MATCH (u)-[r:REVIEWED]->(other:Product)
            WHERE other.asin <> $asin
 
            // Count how many users reviewed both (the collaborative signal)
            WITH other, count(DISTINCT u) AS shared_reviewers
            WHERE shared_reviewers >= 2
 
            // Get product details
            OPTIONAL MATCH (other)-[:MADE_BY]->(b:Brand)
 
            RETURN
                other.asin           AS asin,
                other.title          AS title,
                b.name               AS brand,
                other.average_rating AS avg_rating,
                other.price          AS price,
                shared_reviewers
 
            ORDER BY shared_reviewers DESC
            LIMIT $top_k
        """, {"asin": asin, "top_k": top_k})
        return self._show(results)

    #-----------2.TRENDING---------------------

    def trending(self,category:str,top_k:int=10) -> pd.DataFrame:
        """
                Trending products in a category.

                Idea: Rank products by a simple score that rewards both
                high ratings AND a large number of reviews.

                Score = avg_rating × log(review_count + 1)

                This prevents a product with just 1 five-star review from
                ranking above a product with 4.5 stars and 500 reviews.
                """
        print(f"\n🔥 Trending in: '{category}'\n")
        results = self._query("""
            // Match products in the given category
            MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
            WHERE toLower(c.name) CONTAINS toLower($category)
              AND p.rating_number  >= 5
              AND p.average_rating >= 3.5
 
            OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
 
            // Trending score: balances rating with popularity
            WITH p, b, c,
                 round(p.average_rating * log(toFloat(p.rating_number) + 1), 2)
                 AS trending_score
 
            RETURN
                p.asin           AS asin,
                p.title          AS title,
                b.name           AS brand,
                p.average_rating AS avg_rating,
                p.rating_number  AS review_count,
                p.price          AS price,
                trending_score
 
            ORDER BY trending_score DESC
            LIMIT $top_k
        """, {"category": category, "top_k": top_k})
        if not results:
            print(f"No results for {category}")
            return pd.DataFrame()
        return self._show(results)
    #---------------3.BLEND---------------------------
    def blended(self,asin:str,top_k:int=10) -> pd.DataFrame:
        """
                Blended recommendations — mix of collaborative + trending.

                Steps:
                  1. Get collaborative results for the given product
                  2. Find the product's category, get trending products there
                  3. Merge both lists, remove duplicates
                  4. Sort by a combined score (collab signal + trending signal)
                """
        print(f"\n🎯 Blended Recommendations → ASIN: {asin}\n")

        # Step 1: collaborative results
        collab = self._query("""
            MATCH (u:User)-[:REVIEWED]->(source:Product {asin: $asin})
            MATCH (u)-[:REVIEWED]->(other:Product)
            WHERE other.asin <> $asin
            WITH other, count(DISTINCT u) AS shared_reviewers
            WHERE shared_reviewers >= 2
            OPTIONAL MATCH (other)-[:MADE_BY]->(b:Brand)
            RETURN
                other.asin           AS asin,
                other.title          AS title,
                b.name               AS brand,
                other.average_rating AS avg_rating,
                other.price          AS price,
                shared_reviewers,
                0.0                  AS trending_score
            ORDER BY shared_reviewers DESC
            LIMIT $top_k
        """, {"asin": asin, "top_k": top_k})
        # Step 2: get the product's category than fetch trending
        cat = self._query("""
                   MATCH (p:Product {asin: $asin})-[:BELONGS_TO]->(c:Category)
                   RETURN c.name AS category LIMIT 1
               """, {"asin": asin})

        trending = []
        if cat:
            category = cat[0]["category"]
            trending = self._query("""
                        MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category})
                        WHERE p.asin <> $asin
                          AND p.rating_number  >= 5
                          AND p.average_rating >= 3.5
                        OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
                        WITH p, b,
                             round(p.average_rating * log(toFloat(p.rating_number) + 1), 2)
                             AS trending_score
                        RETURN
                            p.asin           AS asin,
                            p.title          AS title,
                            b.name           AS brand,
                            p.average_rating AS avg_rating,
                            p.price          AS price,
                            0                AS shared_reviewers,
                            trending_score
                        ORDER BY trending_score DESC
                        LIMIT $top_k
                    """, {"asin": asin, "category": category, "top_k": top_k})
            # Step - 3: merge,duplicate, and compute a simple blended score
            seen = {}
            for p in collab:
                seen[p["asin"]] = {**p, "source": "collaborative"}
            for p in trending:
                if p["asin"] not in seen:
                    seen[p["asin"]] = {**p, "source": "trending"}
                else:
                    # product appears in both — boost its score
                    seen[p["asin"]]["trending_score"] = p["trending_score"]
                    seen[p["asin"]]["source"] = "both ✓"

            # Normalize and blend (60% collab, 40% trending)
            all_items = list(seen.values())
            max_collab = max((p["shared_reviewers"] or 0 for p in all_items), default=1)
            max_trending = max((p["trending_score"] or 0 for p in all_items), default=1)

            for p in all_items:
                collab_norm = (p["shared_reviewers"] or 0) / max_collab
                trending_norm = (p["trending_score"] or 0) / max_trending
                p["blended_score"] = round(collab_norm * 0.6 + trending_norm * 0.4, 3)

            all_items.sort(key=lambda x: x["blended_score"], reverse=True)

            return self._show(all_items[:top_k])

    #--------UTILITY---------------------------
    def find_categories(self,keyword:str="") -> pd.DataFrame:
        """
               Search category names — helpful for finding the right name to
               pass into trending().

               Example:
                   engine.find_categories("head")
                   → Headphones, On-Ear Headphones, ...
        """
        results = self._query("""
                   MATCH (c:Category)
                   WHERE toLower(c.name) CONTAINS toLower($keyword)
                   RETURN DISTINCT c.name AS category
                   ORDER BY c.name
                   LIMIT 20
               """, {"keyword": keyword})
        return self._show(results)




engine = RecommendationEngine(uri=NEO4J_URI,user=NEO4J_USER,password=NEO4J_PASSWORD)
#engine.collaborative_filtering("B007U1YGOS")
#engine.trending("ElectronicsCamera & PhotoVideo Surveillance")
#engine.find_categories("head")











