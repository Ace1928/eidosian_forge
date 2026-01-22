import time
import logging
from functools import singledispatchmethod
from itertools import zip_longest
from collections import defaultdict
import rustworkx
from qiskit.circuit import (
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.equivalence import Key, NodeData
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
def edge_cost(self, edge_data):
    """Returns the cost of an edge.

        This function computes the cost of this edge rule by summing
        the costs of all gates in the rule equivalence circuit. In the
        end, we need to subtract the cost of the source since `dijkstra`
        will later add it.
        """
    if edge_data is None:
        return 1
    cost_tot = 0
    for instruction in edge_data.rule.circuit:
        key = Key(name=instruction.operation.name, num_qubits=len(instruction.qubits))
        cost_tot += self._opt_cost_map[key]
    return cost_tot - self._opt_cost_map[edge_data.source]