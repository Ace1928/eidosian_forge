import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
@pytest.mark.parametrize(('vertical_chains', 'ascii_only'), tuple([(vertical_chains, ascii_only) for vertical_chains in [0, 1] for ascii_only in [0, 1]]))
def test_network_text_round_trip(vertical_chains, ascii_only):
    """
    Write the graph to network text format, then parse it back in, assert it is
    the same as the original graph. Passing this test is strong validation of
    both the format generator and parser.
    """
    from networkx.readwrite.text import _parse_network_text
    for graph in generate_test_graphs():
        graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes})
        lines = list(nx.generate_network_text(graph, vertical_chains=vertical_chains, ascii_only=ascii_only))
        new = _parse_network_text(lines)
        try:
            assert new.nodes == graph.nodes
            assert new.edges == graph.edges
        except Exception:
            print('ERROR in round trip with graph')
            nx.write_network_text(graph)
            raise