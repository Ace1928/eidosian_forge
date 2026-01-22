import itertools
from typing import Iterable, Tuple, Dict
import networkx as nx
import cirq
def get_grid_device_graph(*args, **kwargs) -> nx.Graph:
    """Gets the graph of a grid of qubits.

    See GridQubit.rect for argument details."""
    return gridqubits_to_graph_device(cirq.GridQubit.rect(*args, **kwargs))