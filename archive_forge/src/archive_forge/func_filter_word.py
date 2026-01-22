import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32
def filter_word(text):
    """
    Take out english stopwords, punctuation, and compound endings.
    """
    text = normalize(text)
    if regex.match('^\\p{P}+$', text):
        return True
    if text.lower() in STOPWORDS:
        return True
    return False