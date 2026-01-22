from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq
def getGenealogy(self, individual, max_depth=float('inf')):
    """Provide the genealogy tree of an *individual*. The individual must
        have an attribute :attr:`history_index` as defined by
        :func:`~deap.tools.History.update` in order to retrieve its associated
        genealogy tree. The returned graph contains the parents up to
        *max_depth* variations before this individual. If not provided
        the maximum depth is up to the beginning of the evolution.

        :param individual: The individual at the root of the genealogy tree.
        :param max_depth: The approximate maximum distance between the root
                          (individual) and the leaves (parents), optional.
        :returns: A dictionary where each key is an individual index and the
                  values are a tuple corresponding to the index of the parents.
        """
    gtree = {}
    visited = set()

    def genealogy(index, depth):
        if index not in self.genealogy_tree:
            return
        depth += 1
        if depth > max_depth:
            return
        parent_indices = self.genealogy_tree[index]
        gtree[index] = parent_indices
        for ind in parent_indices:
            if ind not in visited:
                genealogy(ind, depth)
            visited.add(ind)
    genealogy(individual.history_index, 0)
    return gtree