import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def contains_qmark(utt, history):
    """
    Sentence-level attribute function.

    See explanation above. Returns 1 if utt contains a question mark, otherwise 0.
    """
    return int('?' in utt)