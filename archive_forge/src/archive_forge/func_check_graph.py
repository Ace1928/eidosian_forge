import pytest
import networkx as nx
from networkx.algorithms.planarity import (
@staticmethod
def check_graph(G, is_planar=None):
    """Raises an exception if the lr_planarity check returns a wrong result

        Parameters
        ----------
        G : NetworkX graph
        is_planar : bool
            The expected result of the planarity check.
            If set to None only counter example or embedding are verified.

        """
    is_planar_lr, result = nx.check_planarity(G, True)
    is_planar_lr_rec, result_rec = check_planarity_recursive(G, True)
    if is_planar is not None:
        if is_planar:
            msg = 'Wrong planarity check result. Should be planar.'
        else:
            msg = 'Wrong planarity check result. Should be non-planar.'
        assert is_planar == is_planar_lr, msg
        assert is_planar == is_planar_lr_rec, msg
    if is_planar_lr:
        check_embedding(G, result)
        check_embedding(G, result_rec)
    else:
        check_counterexample(G, result)
        check_counterexample(G, result_rec)