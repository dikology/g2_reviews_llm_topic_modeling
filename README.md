# G2 Review clustering using LLMs

This project is a proof-of-concept demonstraing how you can use LLMs to perform competitive intelligence on customer reviews and feedback.

In this scenario, we're taking G2 reviews and performing topic modelling in a simple streamlit app.

The overall (processing) pipeline is as follows:
1. Get the G2 company reviews for your target companies (manual step, instructions below) 
2. Basic data reshaping from resulting json (`preprocess.py`)
3. Split reviews into sentences
4. Embed sentences
5. Reduce dimensionality (slightly) and cluster sentences
6. Find N points close to the center of each cluster and stuff them in the LLM to extract the topics
7. Reduce dimensions to 2D in order to visualize

## Getting Started

### Prerequisites


##### Rename `.env.example` to  `.env`.
- Setting the `OPENAI_API_KEY` is mandatory
- If you want to fetch a new set of companies you need to set `APIFY_API_TOKEN`, otherwise, it will use the sample G2 reviews in the repo.



### Installation

#### 1. Clone the repository:
```bash
   git clone https://github.com/balmasi/g2_reviews_llm_topic_modeling
```


#### 2. Create a virtual environment
The easiest way to do this is to use [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
```bash
# Create the g2_reviews_topic_modeling_llm virtual environment
conda create -n g2_reviews_topic_modeling_llm python=3.10
# Activate the virtual environment
conda activate g2_reviews_topic_modelling_llm
```

#### 3. Install the required dependencies: 

```
pip install -r requirements.txt
```


### Getting the G2 Company reviews
1. Browse to your target G2 profiles to grab the slug from the url. For example `https://www.g2.com/products/vena/reviews` would be `vena`
2. Place each target company on a line in the `data/slugs-to-fetch.txt` file. 
3. Set the `APIFY_API_TOKEN` in the .env file to your Apify API token
4. run the create_dataset.py command using `python data/create_dataset.py`


### Running the App

To run the app, execute the following command:

```
streamlit run streamlit_app.py
```

This will start the Streamlit server and launch the app in your default web browser.

```streamlit_chunk.py
# ... existing code ...

import os
import pandas as pd

# Define the path to the CSV file
csv_file_path = "data/reduce_dim_df.csv"

# Check if the CSV file exists
if os.path.exists(csv_file_path):
    # Load the existing data from the CSV file
    reduce_dim_df = pd.read_csv(csv_file_path)
else:
    # Explode the sentences of that review type
    with st.spinner("Parsing review sentences..."):
        xpl_df = explode_reviews(df_cleaned, REVIEW_COL)

    # Embed reviews
    with st.spinner("Vectorizing Reviews..."):
        embedded_df = embed_reviews(xpl_df, REVIEW_COL)

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

    # Save the reduce_dim_df to CSV for future use
    reduce_dim_df.to_csv(csv_file_path, index=False)

# ... existing code ...
fig_clusters = visualize_embeddings(
    reduce_dim_df,
    coords_col="dims_2d",
    review_text_column=REVIEW_COL,
    colour_by_column="cluster_label",
)

st.plotly_chart(fig_clusters, use_container_width=True)

# ... existing code ...