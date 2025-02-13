import textwrap

import streamlit as st

from langchain_core.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain_gigachat.chat_models import GigaChat
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.constants import GIGACHAT_KEY

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
# https://developers.sber.ru/docs/ru/gigachain/get-started/quickstart

model = GigaChat(
    credentials=GIGACHAT_KEY,
    scope="GIGACHAT_API_PERS",
    model="GigaChat",
    streaming=False,
    verify_ssl_certs=False,
)


@st.cache_data(show_spinner=False)
def summarize_cluster(_llm: GigaChat, texts: list):
    """
    Generates a summary label for a cluster of customer reviews.

    Args:
        _llm (GigaChat): The expert summarizer model.
        texts (list): A list of customer reviews.

    Returns:
        str: The generated summary label for the cluster of reviews.
    """
    # Use a cheaper model for the map part

    summarize_one_prompt = textwrap.dedent(
        """
        You are an expert summarizer with the ability to find patterns in a set
        of customer reviews and summarize them into a single concise label.
        Provide a single short (3-10 words) label that encapsulate
        the key points the reviews have in common.
        The label(s) you provide should not be longer than a few words.
        Ensure the label generated is not too vague.

        The reviews are enclosed in triple backticks (```).

        ---
        EXAMPLE 1

        REVIEWS:
        ```
        Review 1: The UI seems to be a little buggy and slow to respond,
        but it's been getting better
        Review 2: I think they could use more integrations.
        The user interface also could use some love. It's finicky and and confusing.
        Review 3: The app user experience needs to be improved.
        It's extremely hard to use.
        ```

        LABEL: UI is hard to use

        ---
        EXAMPLE 2

        REVIEWS:
        ```
        Review 1: The initial price point is pretty high.
        Review 2: Licensing can be a pain in the neck.
        Review 3: Pricing can be lower to favor lower market segments.
        Review 4: The pricing model needs to be simplified.
        ```

        LABEL: Expensive and Confusing Pricing Model
        ---

        REVIEWS:
        ```
        {reviews_text}
        ```

        LABEL:
        """
    )

    prompt = ChatPromptTemplate.from_template(summarize_one_prompt)
    stuffed_reviews_txt = "\n\n".join(
        [f"Review {i}: {txt}" for i, txt in enumerate(texts)]
    )
    chain = prompt | _llm | StrOutputParser()
    return chain.invoke({"reviews_text": stuffed_reviews_txt})


def summarize_sequential(top_n_cluster):
    """
    Generate a summary for each cluster in a top N cluster dictionary.

    Parameters:
        top_n_cluster (dict): A dictionary containing the top N clusters and
        their associated data.
        review_type (str): The type of reviews to summarize
        ('Likes', 'Dislikes', or 'Use-case').

    Returns:
        dict: A dictionary containing the top N clusters with their associated
        data and cluster labels.
    """
    # Limit top_n_cluster to 1 for testing purposes
    # top_n_cluster = dict(
    #     list(top_n_cluster.items())[:1]
    # )

    # Creating a progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    num_clusters = len(top_n_cluster)
    for i, (cluster_id, val) in enumerate(top_n_cluster.items()):
        if cluster_id == -1:
            top_n_cluster[-1]["cluster_label"] = "Uncategorized"
        else:
            top_n_cluster[cluster_id]["cluster_label"] = summarize_cluster(
                model, val["texts"]
            )

        progress = (i + 1) / num_clusters
        progress_bar.progress(progress)
        progress_text.text(f"Naming cluster {i + 1}/{num_clusters}")

    # Ensure the progress bar is full upon completion
    progress_bar.empty()
    progress_text.empty()

    return top_n_cluster
