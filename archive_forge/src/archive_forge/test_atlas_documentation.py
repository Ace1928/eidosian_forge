from itertools import groupby
import pytest
import networkx as nx
from networkx import graph_atlas, graph_atlas_g
from networkx.generators.atlas import NUM_GRAPHS
from networkx.utils import edges_equal, nodes_equal, pairwise
Unit tests for the :func:`~networkx.graph_atlas_g` function.