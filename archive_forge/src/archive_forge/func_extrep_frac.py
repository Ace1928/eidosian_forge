import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def extrep_frac(lst1, lst2):
    """
    Returns the fraction of items in lst1 that are in lst2.
    """
    if len(lst1) == 0:
        return 0
    num_rep = len([x for x in lst1 if x in lst2])
    return num_rep / len(lst1)