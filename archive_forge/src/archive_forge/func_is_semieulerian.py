from itertools import combinations
import networkx as nx
from ..utils import arbitrary_element, not_implemented_for
@nx._dispatch
def is_semieulerian(G):
    """Return True iff `G` is semi-Eulerian.

    G is semi-Eulerian if it has an Eulerian path but no Eulerian circuit.

    See Also
    --------
    has_eulerian_path
    is_eulerian
    """
    return has_eulerian_path(G) and (not is_eulerian(G))