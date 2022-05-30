import re
import nltk


def normalize_document(doc):
    stop_words = nltk.corpus.stopwords.words("english")
    wpt = nltk.WordPunctTokenizer()
    doc = re.sub(r"[^\w\s]", "", doc)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = " ".join(filtered_tokens)
    return doc
