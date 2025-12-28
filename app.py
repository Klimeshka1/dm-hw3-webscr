import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="HW3 Web Scraping App", layout="wide")

# ----------------------------
# ✅ Load CSV data
# ----------------------------
@st.cache_data
def load_data():
    products = pd.read_csv("data/products.csv")
    testimonials = pd.read_csv("data/testimonials.csv")
    reviews = pd.read_csv("data/reviews.csv")

    # convert date column if exists
    if "date" in reviews.columns:
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    return products, testimonials, reviews


products_df, testimonials_df, reviews_df = load_data()

# ----------------------------
# ✅ Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose Section:", ["Products", "Testimonials", "Reviews"])

# ----------------------------
# ✅ Products Section
# ----------------------------
if section == "Products":
    st.title("📦 Products")
    st.write("Showing scraped product data from web-scraping.dev")

    st.dataframe(products_df, use_container_width=True)

# ----------------------------
# ✅ Testimonials Section
# ----------------------------
elif section == "Testimonials":
    st.title("💬 Testimonials")
    st.write("Showing scraped testimonials data from web-scraping.dev")

    st.dataframe(testimonials_df, use_container_width=True)

# ----------------------------
# ✅ Reviews Section (Core Feature)
# ----------------------------
elif section == "Reviews":
    st.title("⭐ Reviews + Sentiment Analysis (2023)")

    # ----------------------------
    # ✅ Month Slider (Jan 2023 - Dec 2023)
    # ----------------------------
    month_list = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
    month_labels = [m.strftime("%b %Y") for m in month_list]

    selected_month = st.select_slider(
        "Select month in 2023:",
        options=month_labels,
        value="Jan 2023"
    )

    # Convert back to datetime
    selected_index = month_labels.index(selected_month)
    selected_date = month_list[selected_index]

    # ----------------------------
    # ✅ Filter Reviews for Selected Month
    # ----------------------------
    filtered = reviews_df[
        (reviews_df["date"].dt.year == 2023) &
        (reviews_df["date"].dt.month == selected_date.month)
    ]

    st.subheader(f"Reviews for: {selected_month}")
    st.write(f"Total reviews in this month: **{len(filtered)}**")

    if filtered.empty:
        st.warning("No reviews found for this month.")
        st.stop()

    st.dataframe(filtered, use_container_width=True)

    # ----------------------------
    # ✅ Sentiment Analysis (Hugging Face)
    # ----------------------------
    st.subheader("Sentiment Analysis Results")

    @st.cache_resource
    def load_model():
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    sentiment_model = load_model()

    # Apply sentiment analysis
    texts = filtered["text"].astype(str).tolist()
    results = sentiment_model(texts)

    filtered = filtered.copy()
    filtered["sentiment"] = [r["label"] for r in results]
    filtered["confidence"] = [r["score"] for r in results]

    st.dataframe(filtered, use_container_width=True)

    # ----------------------------
    # ✅ Visualization: Bar Chart + Average Confidence
    # ----------------------------
    sentiment_counts = filtered["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    avg_conf = filtered.groupby("sentiment")["confidence"].mean().reset_index()
    avg_conf.columns = ["sentiment", "avg_confidence"]

    merged = pd.merge(sentiment_counts, avg_conf, on="sentiment")

    fig = px.bar(
        merged,
        x="sentiment",
        y="count",
        text="count",
        hover_data={"avg_confidence": ":.3f"},
        title="Positive vs Negative Reviews"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("### Average Confidence Score (by Sentiment)")
    st.dataframe(merged, use_container_width=True)
