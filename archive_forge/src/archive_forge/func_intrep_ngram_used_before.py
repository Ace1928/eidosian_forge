import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def intrep_ngram_used_before(dict, hypothesis, history, wt, feat, n):
    """
    Weighted decoding feature function. See explanation above. This feature is 1 for
    words that, if added to the hypothesis, will create a n-gram that has already
    appeared in the hypothesis; otherwise 0.

    Additional inputs:
      n: int, the size of the n-grams considered.
    """
    if hypothesis is not None:
        bad_words = matching_ngram_completions(hypothesis, hypothesis, n)
        if len(bad_words) > 0:
            feat[bad_words] += wt
    return feat