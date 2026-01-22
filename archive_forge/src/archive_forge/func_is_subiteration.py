import pytest
import networkx as nx
from networkx.generators import directed
def is_subiteration(a, b):
    """
    returns True if that each hash sequence in 'a' is a prefix for
    the corresponding sequence indexed by the same node in 'b'.
    """
    return all((b[node][:len(hashes)] == hashes for node, hashes in a.items()))