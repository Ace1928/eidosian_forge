import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
def shortest_path_distance(self, other, simulate_root=False):
    """
        Returns the distance of the shortest path linking the two synsets (if
        one exists). For each synset, all the ancestor nodes and their
        distances are recorded and compared. The ancestor node common to both
        synsets that can be reached with the minimum number of traversals is
        used. If no ancestor nodes are common, None is returned. If a node is
        compared with itself 0 is returned.

        :type other: Synset
        :param other: The Synset to which the shortest path will be found.
        :return: The number of edges in the shortest path connecting the two
            nodes, or None if no path exists.
        """
    if self == other:
        return 0
    dist_dict1 = self._shortest_hypernym_paths(simulate_root)
    dist_dict2 = other._shortest_hypernym_paths(simulate_root)
    inf = float('inf')
    path_distance = inf
    for synset, d1 in dist_dict1.items():
        d2 = dist_dict2.get(synset, inf)
        path_distance = min(path_distance, d1 + d2)
    return None if math.isinf(path_distance) else path_distance