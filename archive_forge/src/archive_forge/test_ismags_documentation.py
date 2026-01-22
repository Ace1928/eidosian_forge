import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso

        For some small, directed, symmetric graphs, make sure that 1) they are
        isomorphic to themselves, and 2) that only the identity mapping is
        found.
        