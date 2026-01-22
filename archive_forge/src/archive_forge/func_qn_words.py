import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def qn_words(dict, hypothesis, history, wt, feat):
    """
    Weighted decoding feature function.

    See explanation above. This feature is 1 for 'interrogative words', 0 otherwise.
    """
    qn_indices = [dict[w] for w in QN_WORDS]
    feat[qn_indices] += wt
    return feat