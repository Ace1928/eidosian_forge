from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def collect_matching_block(self, filter_fn):
    """Iteratively collects the largest block of input nodes (that is, nodes with
        ``_in_degree`` equal to 0) that match a given filtering function.
        Examples of this include collecting blocks of swap gates,
        blocks of linear gates (CXs and SWAPs), blocks of Clifford gates, blocks of single-qubit gates,
        blocks of two-qubit gates, etc.  Here 'iteratively' means that once a node is collected,
        the ``_in_degree`` of each of its immediate successor is decreased by 1, allowing more nodes
        to become input and to be eligible for collecting into the current block.
        Returns the block of collected nodes.
        """
    current_block = []
    unprocessed_pending_nodes = self._pending_nodes
    self._pending_nodes = []
    while unprocessed_pending_nodes:
        new_pending_nodes = []
        for node in unprocessed_pending_nodes:
            if filter_fn(node):
                current_block.append(node)
                for suc in self._direct_succs(node):
                    self._in_degree[suc] -= 1
                    if self._in_degree[suc] == 0:
                        new_pending_nodes.append(suc)
            else:
                self._pending_nodes.append(node)
        unprocessed_pending_nodes = new_pending_nodes
    return current_block