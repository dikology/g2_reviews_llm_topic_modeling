import streamlit as st
import pandas as pd

from src.visualize import visualize_embeddings  # , plot_over_time
import ast

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from collections import defaultdict
import plotly.graph_objects as go

# from src.ui import radio_filter, range_filter

# Ensure nltk resources are available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words("russian")

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


def clean_text(text, stopwords, lemmatizer):
    text = re.sub(
        r"[^a-zA-Zа-яА-Я0-9\\s]", " ", text
    )  # Remove punctuation and special characters
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords)
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize
    return " ".join(words)


reduce_dim_df["cleaned_comment"] = reduce_dim_df["comment"].apply(
    lambda x: clean_text(str(x), stopwords, lemmatizer)
)


def generate_plot_ngrams(
    comments,
    stopwords,
    top_n=12,
    title="Топ 12 биграм и триграм в комментариях",
    return_data=False,
):
    """
    Generate a Plotly bar chart showing the top n biframs (bigrams) and trigrams
    in the provided comments.
    If return_data is True, this function returns tuples (df_bigrams, df_trigrams)
    containing the count of each n-gram.
    """

    def generate_ngrams(text, stopwords, n_gram):
        token = [
            w.lower()
            for sent in nltk.sent_tokenize(text)
            for w in nltk.word_tokenize(sent)
        ]
        # Remove tokens that do not contain any cyrillic characters
        # #and filter out stopwords
        token = [t for t in token if re.search("[а-яА-Я]", t) and t not in stopwords]
        ngrams = zip(*[token[i:] for i in range(n_gram)])
        return [" ".join(ngram) for ngram in ngrams]

    # Count bigrams
    freq_dict = defaultdict(int)
    for sent in comments:
        for ngram in generate_ngrams(sent, stopwords, 2):
            freq_dict[ngram] += 1

    bigram_df = pd.DataFrame(
        sorted(freq_dict.items(), key=lambda x: x[1], reverse=True),
        columns=["ngram", "ncount"],
    )

    # Count trigrams
    freq_dict = defaultdict(int)
    for sent in comments:
        for ngram in generate_ngrams(sent, stopwords, 3):
            freq_dict[ngram] += 1

    trigram_df = pd.DataFrame(
        sorted(freq_dict.items(), key=lambda x: x[1], reverse=True),
        columns=["ngram", "ncount"],
    )

    # Slice top N n-grams
    bigram_df = bigram_df.sort_values(by="ncount", ascending=False).iloc[:top_n]
    trigram_df = trigram_df.sort_values(by="ncount", ascending=False).iloc[:top_n]

    trace0 = go.Bar(
        y=bigram_df.ngram.values,
        x=bigram_df.ncount.values,
        name="Количество биграм",
        orientation="h",
        marker=dict(color="rgb(49,130,189)"),
    )
    trace1 = go.Bar(
        y=trigram_df.ngram.values,
        x=trigram_df.ncount.values,
        name="Количество триграм",
        orientation="h",
        marker=dict(color="rgb(204,204,204)"),
        xaxis="x2",
        yaxis="y2",
    )

    layout = go.Layout(
        height=600,
        width=1200,
        title=title,
        margin=dict(l=150, r=10, t=100, b=100),
        legend=dict(orientation="h"),
        xaxis=dict(domain=[0, 0.4]),
        xaxis2=dict(domain=[0.6, 1]),
        yaxis2=dict(anchor="x2"),
    )
    data = [trace0, trace1]
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(automargin=True)

    st.plotly_chart(fig, use_container_width=True)

    if return_data:
        return bigram_df, trigram_df
    else:
        return bigram_df


# --- Sidebar for Cluster Filtering ---
selected_cluster = sb.selectbox(
    "Select Cluster Label", sorted(reduce_dim_df["cluster_label"].unique()), index=0
)

top_n = sb.slider("Select number of top n-grams", min_value=5, max_value=20, value=12)

# --- Filter the DataFrame based on the selected cluster label ---
filtered_df = reduce_dim_df[reduce_dim_df["cluster_label"] == selected_cluster]

if filtered_df.empty:
    st.info("No data available for the selected cluster.")
else:
    # Get the cleaned comments for further analysis
    filtered_comments = filtered_df["cleaned_comment"].tolist()

    st.header(f"N-grams Analysis for Cluster {selected_cluster}")

    # Generate the n-grams plot and also retrieve the data for bigrams and trigrams
    bigram_df, trigram_df = generate_plot_ngrams(
        filtered_comments,
        stopwords,
        top_n=top_n,
        title=f"Топ n-грам про ТВ for Cluster {selected_cluster}",
        return_data=True,
    )

    # --- Additional UI: Filter Raw Comments Based on Selected n-gram ---
    st.sidebar.markdown("## Filter Comments by n-gram")
    ngram_type = sb.radio(
        "Choose n-gram type for filtering:", options=["Bigram", "Trigram"]
    )

    if ngram_type == "Bigram":
        ngram_options = bigram_df["ngram"].tolist()
    else:
        ngram_options = trigram_df["ngram"].tolist()

    selected_ngram = sb.selectbox(
        "Select an n-gram to filter raw comments", options=ngram_options
    )

    # Filter raw comments using the cleaned comment as a filter,
    # but display the original raw comment text.
    filtered_raw_comments = filtered_df[
        filtered_df["cleaned_comment"].str.contains(selected_ngram, na=False)
    ]["comment"].tolist()

    st.markdown(f"### Raw Comments Containing '{selected_ngram}'")
    if not filtered_raw_comments:
        st.write("No comments found for the selected n-gram.")
    else:
        for comment in filtered_raw_comments:
            st.write(comment)
