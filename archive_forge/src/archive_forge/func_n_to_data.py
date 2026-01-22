from itertools import islice
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for, open_file
def n_to_data(n):
    """Convert an integer to one-, four- or eight-unit graph6 sequence.

    This function is undefined if `n` is not in ``range(2 ** 36)``.

    """
    if n <= 62:
        return [n]
    elif n <= 258047:
        return [63, n >> 12 & 63, n >> 6 & 63, n & 63]
    else:
        return [63, 63, n >> 30 & 63, n >> 24 & 63, n >> 18 & 63, n >> 12 & 63, n >> 6 & 63, n & 63]