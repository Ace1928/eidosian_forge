from typing import TYPE_CHECKING
import networkx as nx
from cirq import devices, ops
def construct_ring_device(l: int, directed: bool=False) -> RoutingTestingDevice:
    nx_graph = nx.cycle_graph(l, create_using=nx.DiGraph if directed else nx.Graph)
    return RoutingTestingDevice(nx_graph)