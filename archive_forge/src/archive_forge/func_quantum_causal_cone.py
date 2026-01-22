from node A to node B means that the (qu)bit passes from the output of A
from collections import OrderedDict, defaultdict, deque, namedtuple
import copy
import math
from typing import Dict, Generator, Any, List
import numpy as np
import rustworkx as rx
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources, node_resources, CONTROL_FLOW_OP_NAMES
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit.bit import Bit
def quantum_causal_cone(self, qubit):
    """
        Returns causal cone of a qubit.

        A qubit's causal cone is the set of qubits that can influence the output of that
        qubit through interactions, whether through multi-qubit gates or operations. Knowing
        the causal cone of a qubit can be useful when debugging faulty circuits, as it can
        help identify which wire(s) may be causing the problem.

        This method does not consider any classical data dependency in the ``DAGCircuit``,
        classical bit wires are ignored for the purposes of building the causal cone.

        Args:
            qubit (~qiskit.circuit.Qubit): The output qubit for which we want to find the causal cone.

        Returns:
            Set[~qiskit.circuit.Qubit]: The set of qubits whose interactions affect ``qubit``.
        """
    output_node = self.output_map.get(qubit, None)
    if not output_node:
        raise DAGCircuitError(f'Qubit {qubit} is not part of this circuit.')
    qubits_to_check = {qubit}
    queue = deque(self.predecessors(output_node))
    while queue:
        node_to_check = queue.popleft()
        if isinstance(node_to_check, DAGOpNode):
            qubit_set = set(node_to_check.qargs)
            if len(qubit_set.intersection(qubits_to_check)) > 0 and node_to_check.op.name != 'barrier' and (not getattr(node_to_check.op, '_directive')):
                qubits_to_check = qubits_to_check.union(qubit_set)
        for node in self.quantum_predecessors(node_to_check):
            if isinstance(node, DAGOpNode) and len(qubits_to_check.intersection(set(node.qargs))) > 0:
                queue.append(node)
    return qubits_to_check