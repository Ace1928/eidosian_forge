import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
Test subgraph chains that both restrict and show nodes/edges.

        A restricted_view subgraph should allow induced subgraphs using
        G.subgraph that automagically without a chain (meaning the result
        is a subgraph view of the original graph not a subgraph-of-subgraph.
        