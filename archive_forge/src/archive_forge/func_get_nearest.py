import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
def get_nearest(self, query, top_size=0):
    """
        Get approximate nearest neighbors for query from index.

        Parameters
        ----------
        query : list or numpy.ndarray
            Vector for which nearest neighbors should be found.
        top_size : int
            Required number of neighbors.

        Returns
        -------
        neighbors : list of tuples (id, distance)
        """
    return self._online_index._get_nearest_neighbors(query, top_size)