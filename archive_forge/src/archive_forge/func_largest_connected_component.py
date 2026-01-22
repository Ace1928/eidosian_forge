import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
def largest_connected_component(self):
    """Return a set of qubits in the largest connected component."""
    return max(rx.weakly_connected_components(self.graph), key=len)