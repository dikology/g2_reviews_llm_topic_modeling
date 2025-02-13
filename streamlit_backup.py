import streamlit as st
import pandas as pd

from src.visualize import visualize_embeddings  # , plot_over_time
import ast

# from src.ui import radio_filter, range_filter


REVIEW_COL = "comment"
reduced_dim_csv_path = "data/reduced_dim_reviews.csv"

# Set page to wide mode
st.set_page_config(layout="wide")
sb = st.sidebar

reduce_dim_df = pd.read_csv(reduced_dim_csv_path)
# Convert the string representation of lists back to lists
reduce_dim_df["dims_2d"] = reduce_dim_df["dims_2d"].apply(ast.literal_eval)

fig_clusters = visualize_embeddings(
    reduce_dim_df,
    coords_col="dims_2d",
    review_text_column=REVIEW_COL,
    colour_by_column="cluster_label",
)


st.plotly_chart(fig_clusters, use_container_width=True)


# fig_publish_dates = plot_over_time(filtered_df, "date_published")

# st.plotly_chart(fig_publish_dates, use_container_width=True)
