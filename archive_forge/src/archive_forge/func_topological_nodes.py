import math
import heapq
from collections import OrderedDict, defaultdict
import rustworkx as rx
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.dagcircuit.dagdepnode import DAGDepNode
def topological_nodes(self):
    """
        Yield nodes in topological order.

        Returns:
            generator(DAGNode): node in topological order.
        """

    def _key(x):
        return x.sort_key
    return iter(rx.lexicographical_topological_sort(self._multi_graph, key=_key))