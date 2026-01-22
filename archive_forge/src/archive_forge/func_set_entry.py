import copy
from collections import namedtuple
from rustworkx.visualization import graphviz_draw
import rustworkx as rx
from qiskit.exceptions import InvalidFileError
from .exceptions import CircuitError
from .parameter import Parameter
from .parameterexpression import ParameterExpression
def set_entry(self, gate, entry):
    """Set the equivalence record for a Gate. Future queries for the Gate
        will return only the circuits provided.

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            entry (List['QuantumCircuit']) : A list of QuantumCircuits, each
                equivalently implementing the given Gate.
        """
    for equiv in entry:
        _raise_if_shape_mismatch(gate, equiv)
        _raise_if_param_mismatch(gate.params, equiv.parameters)
    node_index = self._set_default_node(Key(name=gate.name, num_qubits=gate.num_qubits))
    self._graph[node_index].equivs.clear()
    for parent, child, _ in self._graph.in_edges(node_index):
        self._graph.remove_edge(parent, child)
    for equivalence in entry:
        self.add_equivalence(gate, equivalence)