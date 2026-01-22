import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
def get_nearest_and_add_item(self, query):
    """
        Get approximate nearest neighbors for query from index and add item to index

        Parameters
        ----------
        query : list or numpy.ndarray
            Vector for which nearest neighbors should be found.
            Vector which should be added in index.

        Returns
        -------
        neighbors : list of tuples (id, distance) with length = search_neighborhood_size
        """
    return self._online_index._get_nearest_neighbors_and_add_item(query)