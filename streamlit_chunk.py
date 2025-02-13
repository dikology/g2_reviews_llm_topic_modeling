import streamlit as st

from src.preprocess import explode_reviews, preprocess_data
from src.embeddings import embed_reviews, reduce_dimensions_append_array

from src.extract_topic import summarize_sequential
from src.cluster import cluster_and_append, find_closest_to_centroid

from src.visualize import visualize_embeddings  # , plot_over_time

# from src.ui import radio_filter, range_filter


REVIEW_COL = "comment"


# def select_reviews_of_type(df, review_type):
#     if review_type == "Negative":
#         return df[["id", "likes"]].rename(columns={"likes": REVIEW_COL})
#     elif review_type == "Positive":
#         return df[["id", "dislikes"]].rename(columns={"dislikes": REVIEW_COL})
#     elif review_type == "Use-case":
#         return df[["id", "usecase"]].rename(columns={"usecase": REVIEW_COL})
#     else:
#         raise ValueError("Unexpected review type")


# entry point
# preprocess_data returns a dataframe based on a given json.
# I can make it void instead because I'll go for SQL
df_cleaned = preprocess_data()
base_df = df_cleaned[
    [
        "row_id",
        "segment",
        "question",
        "comment",
        "answer",
        "datetime",
    ]
]

# Set page to wide mode
st.set_page_config(layout="wide")
sb = st.sidebar

# Select a company
# company_counts = base_df["product.slug"].value_counts()
# companies_with_counts = {
#     f"{company} ({count})": company for company, count in company_counts.items()
# }
# selected_company_label = sb.selectbox("Company", companies_with_counts.keys())
# selected_company = companies_with_counts[selected_company_label]

# Select a review type
# review_type = sb.radio("Review Type", ["Positive", "Negative", "Neutral"])

# df_of_type = select_reviews_of_type(df_cleaned, review_type)

# Explode the sentences of that review type
with st.spinner("Parsing review sentences..."):
    xpl_df = explode_reviews(df_cleaned, REVIEW_COL)

# Embed reviews
with st.spinner("Vectorizing Reviews..."):
    embedded_df = embed_reviews(xpl_df, REVIEW_COL)

# Filter to selected company
# company_df = base_df[base_df["product.slug"] == selected_company].merge(
#     embedded_df, on="id"
# )

with st.spinner("Clustering Reviews..."):
    clustered_df = cluster_and_append(embedded_df, f"{REVIEW_COL}_embeddings", 15)


NUM_REVIEWS_TO_USE_IN_CLUSTER_LABEL = 30
top_cluster_docs = find_closest_to_centroid(
    clustered_df,
    NUM_REVIEWS_TO_USE_IN_CLUSTER_LABEL,
    f"{REVIEW_COL}_embeddings",
    f"{REVIEW_COL}_embeddings_cluster_id",
    REVIEW_COL,
)
print(f"number of clusters: {len(top_cluster_docs)}")
top_cluster_docs = summarize_sequential(top_cluster_docs)
top_cluster_map = {
    cluster_id: data["cluster_label"] for cluster_id, data in top_cluster_docs.items()
}
clustered_df["cluster_label"] = clustered_df[f"{REVIEW_COL}_embeddings_cluster_id"].map(
    top_cluster_map
)

# # Reduce the embedding space to 2D for visualization
reduce_dim_df = reduce_dimensions_append_array(
    clustered_df, f"{REVIEW_COL}_embeddings", num_dimensions=2, dim_col_name="dims_2d"
)


# # FILTERS
# filtered_df = radio_filter("Source", sb, reduce_dim_df, "source.type")
# filtered_df = radio_filter("Segment", sb, filtered_df, "segment")
# filtered_df = range_filter("Review Date", sb, filtered_df, "date_published")


fig_clusters = visualize_embeddings(
    reduce_dim_df,
    coords_col="dims_2d",
    review_text_column=REVIEW_COL,
    colour_by_column="cluster_label",
)


st.plotly_chart(fig_clusters, use_container_width=True)


# fig_publish_dates = plot_over_time(filtered_df, "date_published")

# st.plotly_chart(fig_publish_dates, use_container_width=True)
