import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def intrep_repeated_word_frac(utt, history, remove_stopwords):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns the fraction of words in utt that are repeated.
    Additional inputs:
      remove_stopwords: bool. If True, stopwords are removed before counting repetition.
    """
    assert utt.strip() != ''
    tokens = utt.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return intrep_frac(tokens)