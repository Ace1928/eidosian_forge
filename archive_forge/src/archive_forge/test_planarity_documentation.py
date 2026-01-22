import pytest
import networkx as nx
from networkx.algorithms.planarity import (
Raises an exception if the lr_planarity check returns a wrong result

        Parameters
        ----------
        G : NetworkX graph
        is_planar : bool
            The expected result of the planarity check.
            If set to None only counter example or embedding are verified.

        