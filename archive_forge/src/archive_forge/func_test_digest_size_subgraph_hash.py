import pytest
import networkx as nx
from networkx.generators import directed
def test_digest_size_subgraph_hash():
    """
    The hash string lengths should be as expected for a variety of graphs and
    digest sizes
    """
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(1, r + 1):
        G = nx.erdos_renyi_graph(n, p * i, seed=1000 + i)
        digest_size16_hashes = nx.weisfeiler_lehman_subgraph_hashes(G)
        digest_size32_hashes = nx.weisfeiler_lehman_subgraph_hashes(G, digest_size=32)
        assert digest_size16_hashes != digest_size32_hashes
        assert hexdigest_sizes_correct(digest_size16_hashes, 16)
        assert hexdigest_sizes_correct(digest_size32_hashes, 32)