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
def get_positive_relation_ranks_and_avg_prec(all_distances, positive_relations):
    """Compute ranks and Average Precision of positive relations.

        Parameters
        ----------
        all_distances : numpy.array of float
            Array of all distances (floats) for a specific item.
        positive_relations : list
            List of indices of positive relations for the item.

        Returns
        -------
        (list of int, float)
            The list contains ranks of positive relations in the same order as `positive_relations`.
            The float is the Average Precision of the ranking, e.g. ([1, 2, 3, 20], 0.610).

        """
    positive_relation_distances = all_distances[positive_relations]
    negative_relation_distances = np.ma.array(all_distances, mask=False)
    negative_relation_distances.mask[positive_relations] = True
    ranks = (negative_relation_distances < positive_relation_distances[:, np.newaxis]).sum(axis=1) + 1
    map_ranks = np.sort(ranks) + np.arange(len(ranks))
    avg_precision = (np.arange(1, len(map_ranks) + 1) / np.sort(map_ranks)).mean()
    return (list(ranks), avg_precision)