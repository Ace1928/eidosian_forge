import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
@staticmethod
def vector_distance(vector_1, vector_2):
    """Compute poincare distance between two input vectors. Convenience method over `vector_distance_batch`.

        Parameters
        ----------
        vector_1 : numpy.array
            Input vector.
        vector_2 : numpy.array
            Input vector.

        Returns
        -------
        numpy.float
            Poincare distance between `vector_1` and `vector_2`.

        """
    return PoincareKeyedVectors.vector_distance_batch(vector_1, vector_2[np.newaxis, :])[0]