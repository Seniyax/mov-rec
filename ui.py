import streamlit as st
import requests
import pandas as pd

API = "http://localhost:8000"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🛒 Product Recommender",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 E-Commerce Recommendation Engine")
st.caption("Powered by Neo4j Knowledge Graph + RAG (Gemini 2.0 Flash)")
st.divider()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def call_api(endpoint: str, params: dict = {}) -> dict | None:
    """Call the FastAPI backend and handle errors gracefully."""
    try:
        res = requests.get(f"{API}{endpoint}", params=params, timeout=60)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Make sure `uvicorn api:app --port 8000` is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


def show_products(results: list[dict]):
    """Display a list of products as Streamlit cards in a 3-column grid."""
    if not results:
        st.info("No products found.")
        return

    cols = st.columns(3)
    for i, p in enumerate(results):
        with cols[i % 3]:
            with st.container(border=True):

                # Title
                title = str(p.get("title") or "Unknown Product")
                st.markdown(f"**{title[:70]}{'...' if len(title) > 70 else ''}**")

                # Brand + rating row
                brand = p.get("brand") or "—"
                rating = p.get("avg_rating")
                stars = f"⭐ {rating}" if rating else ""
                st.caption(f"🏷 {brand}   {stars}")

                # Price
                price = p.get("price")
                if price:
                    st.markdown(f"💰 **${price}**")

                # Signal badges
                source = p.get("source", "")
                if source == "both ✓":
                    st.success("📦 Collab + Trending", icon="✓")
                elif source == "collaborative":
                    shared = p.get("shared_reviewers", "")
                    st.info(f"👥 {shared} shared reviewers")
                elif source == "trending":
                    score = p.get("trending_score", "")
                    st.warning(f"🔥 Trending score: {score}")

                # ASIN
                asin = p.get("asin", "")
                st.code(asin, language=None)


@st.cache_data(ttl=300)
def fetch_categories() -> list[str]:
    """Fetch category list from API — cached for 5 minutes."""
    data = call_api("/categories")
    return data["categories"] if data else []


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Results to show", min_value=3, max_value=15, value=6)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    - 🎯 **Recommendations** use collaborative filtering —
      products reviewed by the same users
    - 🔥 **Trending** ranks by rating × log(review count)
    - 💡 **Explain** sends results to Gemini for a natural language explanation
    """)
    st.divider()
    st.caption("Neo4j · sentence-transformers · Gemini 2.0 Flash · LangChain")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🎯 Recommendations",
    "🔥 Trending",
    "💡 Explain",
])

# ── TAB 1: RECOMMENDATIONS ────────────────────────────────────────────────────

with tab1:
    st.subheader("🎯 Get Product Recommendations")
    st.markdown("Enter a product ASIN to find similar products that other users also reviewed.")

    col1, col2 = st.columns([3, 1])
    with col1:
        asin_input = st.text_input(
            "Product ASIN",
            placeholder="e.g. B09XYZ1234",
            label_visibility="collapsed",
        )
    with col2:
        search_btn = st.button("Get Recommendations", type="primary", use_container_width=True)

    if search_btn:
        if not asin_input.strip():
            st.warning("Please enter a product ASIN.")
        else:
            with st.spinner("Finding recommendations..."):
                data = call_api(f"/recommend/{asin_input.strip()}", {"top_k": top_k})

            if data:
                results = data.get("results", [])
                st.markdown(f"**{len(results)} recommendations for** `{asin_input}`")
                st.divider()
                show_products(results)

    # Helper: show a random valid ASIN from the graph
    with st.expander("🔍 Don't have an ASIN? Fetch one from Neo4j"):
        st.code("""
# Run this in your notebook to get a valid ASIN:
from recommendation_engine import RecommendationEngine
engine = RecommendationEngine(uri=..., user=..., password=...)
result = engine._query(
    "MATCH (p:Product) WHERE p.average_rating >= 4.0 "
    "RETURN p.asin AS asin, p.title AS title LIMIT 5"
)
for r in result: print(r)
        """, language="python")

# ── TAB 2: TRENDING ───────────────────────────────────────────────────────────

with tab2:
    st.subheader("🔥 Trending Products by Category")
    st.markdown("See the highest rated and most reviewed products in any category.")

    categories = fetch_categories()

    col1, col2 = st.columns([3, 1])
    with col1:
        if categories:
            category_input = st.selectbox(
                "Select a category",
                options=categories,
                label_visibility="collapsed",
            )
        else:
            category_input = st.text_input(
                "Category name",
                placeholder="e.g. Headphones",
                label_visibility="collapsed",
            )
    with col2:
        trend_btn = st.button("Show Trending", type="primary", use_container_width=True)

    if trend_btn:
        if not category_input:
            st.warning("Please select or enter a category.")
        else:
            with st.spinner(f"Fetching trending in '{category_input}'..."):
                data = call_api(f"/trending/{category_input}", {"top_k": top_k})

            if data:
                results = data.get("results", [])
                if results:
                    st.markdown(f"**Top {len(results)} trending in** `{category_input}`")
                    st.divider()
                    show_products(results)
                else:
                    st.info(f"No trending products found for '{category_input}'. Try a different category.")

# ── TAB 3: EXPLAIN ────────────────────────────────────────────────────────────

with tab3:
    st.subheader("💡 Explain Recommendations")
    st.markdown(
        "Enter a product ASIN and Gemini will explain **why** similar products "
        "are recommended, based on the knowledge graph."
    )

    st.info(
        "⏱ This uses the Gemini API — may take 5–10 seconds. "
        "If you hit a rate limit, wait ~15 seconds and try again.",
        icon="ℹ️",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        explain_asin = st.text_input(
            "Product ASIN to explain",
            placeholder="e.g. B09XYZ1234",
            key="explain_asin",
            label_visibility="collapsed",
        )
    with col2:
        explain_btn = st.button("Explain", type="primary", use_container_width=True)

    if explain_btn:
        if not explain_asin.strip():
            st.warning("Please enter a product ASIN.")
        else:
            with st.spinner("Generating explanation with Gemini..."):
                data = call_api(f"/explain/{explain_asin.strip()}")

            if data:
                explanation = data.get("explanation", "")
                if explanation:
                    st.divider()
                    st.markdown("### 🤖 Gemini's Explanation")
                    st.markdown(explanation)
                else:
                    st.warning("No explanation returned. The product may not have enough graph data.")