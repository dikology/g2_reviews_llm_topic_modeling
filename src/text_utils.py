# Let's split the reviews into sentences so we can get a fine-grained view

import nltk
from nltk.tokenize import sent_tokenize

# Ensure that the necessary NLTK resources are downloaded
nltk.download("punkt")


def split_into_sentences(text, ngram=1):
    # Use NLTK to split the text into sentences
    sentences = sent_tokenize(text, language="russian")

    # Handle the case where ngram is 1 (default behavior)
    if ngram == 1:
        return sentences

    # Group sentences into n-grams for ngram > 1
    ngrams = [
        " ".join(sentences[i : i + ngram]) for i in range(len(sentences) - ngram + 1)
    ]
    return ngrams
