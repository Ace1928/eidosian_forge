import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
@staticmethod
def log_accuracy(section):
    correct, incorrect = (len(section['correct']), len(section['incorrect']))
    if correct + incorrect > 0:
        logger.info('%s: %.1f%% (%i/%i)', section['section'], 100.0 * correct / (correct + incorrect), correct, correct + incorrect)