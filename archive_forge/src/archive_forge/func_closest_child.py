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
def closest_child(self, node):
    """Get the node closest to `node` that is lower in the hierarchy than `node`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which closest child is to be found.

        Returns
        -------
        {str, None}
            Node closest to `node` that is lower in the hierarchy than `node`.
            If there are no nodes lower in the hierarchy, None is returned.

        """
    all_distances = self.distances(node)
    all_norms = np.linalg.norm(self.vectors, axis=1)
    node_norm = all_norms[self.get_index(node)]
    mask = node_norm >= all_norms
    if mask.all():
        return None
    all_distances = np.ma.array(all_distances, mask=mask)
    closest_child_index = np.ma.argmin(all_distances)
    return self.index_to_key[closest_child_index]