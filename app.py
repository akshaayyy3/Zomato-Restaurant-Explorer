import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="Zomato Restaurant Recommendation", layout="wide")
st.title("🍴 Zomato Restaurant Explorer")

# ------------------------------
# Load Dataset (Optimized & Portable)
# ------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Default local fallback


        csv_path = Path("zomato_dataset.csv")
        df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )

    # Handle duplicates and missing values
    df.drop_duplicates(inplace=True)
    df.fillna(
        {
            "dining_rating": 0,
            "delivery_rating": 0,
            "votes": 0,
            "prices": np.nan,
            "cuisine": "Unknown",
            "place_name": "Unknown",
            "city": "Unknown",
        },
        inplace=True,
    )

    # Type conversions
    for col in ["dining_rating", "delivery_rating", "votes", "prices"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Unified rating
    df["unified_rating"] = df[["dining_rating", "delivery_rating"]].max(axis=1)

    # Fill missing prices with median
    df["prices"].replace(0, np.nan, inplace=True)
    df["prices"].fillna(df["prices"].median(), inplace=True)

    # Price category
    bins = [0, 200, 500, 1000, 2000, np.inf]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    df["price_range_cat"] = pd.cut(df["prices"], bins=bins, labels=labels)

    # Extract primary cuisine
    df["primary_cuisine"] = df["cuisine"].astype(str).apply(lambda x: x.split(",")[0])

    # Normalized scores
    df["norm_votes"] = df["votes"] / (df["votes"].max() or 1)
    df["norm_price_inv"] = 1 - (df["prices"] / (df["prices"].max() or 1))

    # Composite score
    df["rate"] = (
        (df["unified_rating"] / 5.0) * 0.5 + df["norm_votes"] * 0.3 + df["norm_price_inv"] * 0.2
    )
    df["best_seller_score"] = np.where(
        df["best_seller"].astype(str).str.upper() == "BESTSELLER",
        1,
        0
    )
    # ------------------------------
    # Restaurant Level Aggregation
    # ------------------------------
    df = (
        df.groupby(
            ["restaurant_name", "city", "place_name", "primary_cuisine"],
            as_index=False
        )
        .agg({
            "dining_rating": "mean",
            "delivery_rating": "mean",
            "votes": "sum",
            "prices": "mean",
            "rate": "mean",
            "best_seller_score": "max"
        })
    )
    # Recreate price category after aggregation
    bins = [0, 200, 500, 1000, 2000, np.inf]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    df["price_range_cat"] = pd.cut(
        df["prices"],
        bins=bins,
        labels=labels
    )

    return df


uploaded_file = st.sidebar.file_uploader("Upload Zomato CSV", type=["csv"])
df = load_data(uploaded_file)

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("🔍 Filters")

cities = sorted(df["city"].unique())
selected_cities = st.sidebar.multiselect("City", cities, default=cities[:1])

places = (
    sorted(df[df["city"].isin(selected_cities)]["place_name"].unique())
    if selected_cities
    else []
)
selected_places = st.sidebar.multiselect("Place", places, default=places)

cuisines = sorted(df["primary_cuisine"].unique())
selected_cuisines = st.sidebar.multiselect(
    "Cuisine",
    cuisines,
    default=cuisines
)

min_rate = st.sidebar.slider("Minimum Rating", 0.0, 1.0, 0.3, 0.01)
min_votes = st.sidebar.number_input("Minimum Votes", 0, 10000, 50, 10)

price_options = df["price_range_cat"].cat.categories.tolist()
selected_price_range = st.sidebar.multiselect("Price Range", price_options, default=price_options)

num_recommend = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# ------------------------------
# Filtered Dataset
# ------------------------------
filtered_df = df[
    (df["city"].isin(selected_cities))
    & (df["place_name"].isin(selected_places))
    & (df["primary_cuisine"].isin(selected_cuisines))
    & (df["rate"] >= min_rate)
    & (df["votes"] >= min_votes)
    & (df["price_range_cat"].isin(selected_price_range))
].copy()

# ------------------------------
# Dashboard Overview
# ------------------------------

st.markdown("## 📊 Dashboard Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Restaurants", len(filtered_df))

avg_rating = round(filtered_df["rate"].mean(),2) if not filtered_df.empty else 0
c2.metric("Average Rating", avg_rating)

avg_price = int(filtered_df["prices"].mean()) if not filtered_df.empty else 0
c3.metric("Average Price", f"₹{avg_price}")

top_cuisine = (
    filtered_df["primary_cuisine"].mode()[0]
    if not filtered_df.empty else "-"
)

c4.metric("Top Cuisine", top_cuisine)
# ------------------------------
# Recommendation Ranking
# ------------------------------
if not filtered_df.empty:

    def normalize(series):
        if series.max() == series.min():
            return pd.Series(np.ones(len(series)), index=series.index)

        return (series - series.min()) / (series.max() - series.min())

    filtered_df["norm_rate"] = normalize(filtered_df["rate"])
    filtered_df["norm_votes"] = normalize(filtered_df["votes"])
    filtered_df["norm_price"] = normalize(filtered_df["prices"])

    filtered_df["composite_score"] = (
        0.45 * filtered_df["norm_rate"]
        + 0.30 * filtered_df["norm_votes"]
        + 0.15 * (1 - filtered_df["norm_price"])
        + 0.10 * filtered_df["best_seller_score"]
    )
    filtered_df["composite_score"] = (
            filtered_df["composite_score"] * 100
    ).round(2)

    recommendations = (
        filtered_df.sort_values(by="composite_score", ascending=False)
        .head(num_recommend)
        .reset_index(drop=True)
    )
else:
    recommendations = pd.DataFrame()

# ------------------------------
# Display Recommendations
# ------------------------------
st.subheader(f"⭐ Top {num_recommend} Restaurant Recommendations")
if not recommendations.empty:
    st.dataframe(
        recommendations[
            [
                "restaurant_name",
                "city",
                "place_name",
                "primary_cuisine",
                "rate",
                "votes",
                "prices",
                "best_seller_score",
                "composite_score"
            ]
        ].style.background_gradient(
            subset=["composite_score"],
            cmap="Greens",
        ),
        use_container_width=True,
    )

else:

    st.info("No restaurants match your filters. Try relaxing your criteria.")
st.success(f"Found {len(recommendations)} restaurant recommendations.")
# ------------------------------
# Clustering
# ------------------------------
st.sidebar.header("🧠 Clustering")

categorical_cols = ["primary_cuisine", "place_name", "city"]
df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
df_encoded["votes"] = df["votes"]
df_encoded["rate"] = df["rate"]
df_encoded["prices"] = df["prices"]
df_encoded["best_seller"] = df["best_seller_score"]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

max_clusters = min(10, len(df))
n_clusters = st.sidebar.slider("Number of Clusters", 2, max_clusters, 5)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(scaled_data)

st.markdown("---")
st.subheader("📊 Cluster Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cluster Summary")

    summary = (
        df.groupby("cluster")
        .agg(
            Restaurants=("restaurant_name", "count"),
            Avg_Rating=("rate", "mean"),
            Avg_Price=("prices", "mean"),
            Avg_Votes=("votes", "mean")
        )
        .round(2)
    )

    st.dataframe(summary, use_container_width=True)


with col2:
    cluster_ratings = df.groupby("cluster")["rate"].mean()

    fig, ax = plt.subplots(figsize=(7,5))

    cluster_ratings.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Average Rating by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Average Rating")

    st.pyplot(fig)

# ------------------------------
# Dimensionality Reduction (Optional Button for Speed)
# ------------------------------
if st.sidebar.button("Run Dimensionality Reduction (PCA + t-SNE)"):
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(scaled_data)
    df["pca1"], df["pca2"] = pca_data[:, 0], pca_data[:, 1]

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(df) - 1),
        n_iter=500,
    )
    tsne_data = tsne.fit_transform(scaled_data)
    df["tsne1"], df["tsne2"] = tsne_data[:, 0], tsne_data[:, 1]

    st.subheader("PCA Clusters")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x="pca1", y="pca2", hue="cluster", palette="tab10", data=df, s=40, ax=ax1)
    st.pyplot(fig1)

    st.subheader("t-SNE Clusters")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x="tsne1", y="tsne2", hue="cluster", palette="tab10", data=df, s=40, ax=ax2)
    st.pyplot(fig2)

# ------------------------------
# Additional Insights
# ------------------------------
st.markdown("---")
st.subheader("📈 Insights")

col3, col4 = st.columns(2)
with col3:

    st.write("Top 10 Cuisines")

    top = df["primary_cuisine"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        x=top.values,
        y=top.index,
        palette="viridis",
        ax=ax
    )

    ax.set_xlabel("Restaurants")
    ax.set_ylabel("Cuisine")
    ax.set_title("Top 10 Cuisines")

    st.pyplot(fig)
    st.write("Top 10 Restaurants by Votes")

    top_restaurants = (
        df.sort_values("votes", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=top_restaurants,
        x="votes",
        y="restaurant_name",
        palette="rocket",
        ax=ax
    )

    ax.set_title("Most Popular Restaurants")

    st.pyplot(fig)
with col4:

    st.write("Restaurant Price Distribution")

    fig3, ax3 = plt.subplots(figsize=(8,5))

    sns.histplot(
        data=df,
        x="prices",
        bins=30,
        kde=True,
        color="royalblue",
        ax=ax3
    )

    ax3.set_title("Restaurant Price Distribution")
    ax3.set_xlabel("Average Cost")
    ax3.set_ylabel("Restaurant Count")

    st.pyplot(fig3)

# ------------------------------
# Export Recommendations
# ------------------------------
if not recommendations.empty:
    st.download_button(
        label="💾 Download Recommendations (CSV)",
        data=recommendations.to_csv(index=False).encode("utf-8"),
        file_name="zomato_recommendations.csv",
        mime="text/csv",
    )
