import networkx as nx
from networkx.exception import NetworkXError
from networkx.readwrite.graph6 import data_to_n, n_to_data
from networkx.utils import not_implemented_for, open_file
def enc(x):
    """Big endian k-bit encoding of x"""
    return [1 if x & 1 << k - 1 - i else 0 for i in range(k)]