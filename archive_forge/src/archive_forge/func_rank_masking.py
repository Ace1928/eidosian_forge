import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
def rank_masking(a, threshold=None):
    """Faster masking method. Returns a new binary mask."""
    if threshold is None:
        threshold = 0.11
    return a > np.sort(a)[::-1][int(len(a) * threshold)]