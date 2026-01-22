import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def largest_common_subgraph(self, symmetry=True):
    """
        Find the largest common induced subgraphs between :attr:`subgraph` and
        :attr:`graph`.

        Parameters
        ----------
        symmetry: bool
            Whether symmetry should be taken into account. If False, found
            largest common subgraphs may be symmetrically equivalent.

        Yields
        ------
        dict
            The found isomorphism mappings of {graph_node: subgraph_node}.
        """
    if not self.subgraph:
        yield {}
        return
    elif not self.graph:
        return
    if symmetry:
        _, cosets = self.analyze_symmetry(self.subgraph, self._sgn_partitions, self._sge_colors)
        constraints = self._make_constraints(cosets)
    else:
        constraints = []
    candidates = self._find_nodecolor_candidates()
    if any(candidates.values()):
        yield from self._largest_common_subgraph(candidates, constraints)
    else:
        return