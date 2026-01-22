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
def remove_clbits(self, *clbits):
    """
        Remove classical bits from the circuit. All bits MUST be idle.
        Any registers with references to at least one of the specified bits will
        also be removed.

        Args:
            clbits (List[Clbit]): The bits to remove.

        Raises:
            DAGCircuitError: a clbit is not a :obj:`.Clbit`, is not in the circuit,
                or is not idle.
        """
    if any((not isinstance(clbit, Clbit) for clbit in clbits)):
        raise DAGCircuitError('clbits not of type Clbit: %s' % [b for b in clbits if not isinstance(b, Clbit)])
    clbits = set(clbits)
    unknown_clbits = clbits.difference(self.clbits)
    if unknown_clbits:
        raise DAGCircuitError('clbits not in circuit: %s' % unknown_clbits)
    busy_clbits = {bit for bit in clbits if not self._is_wire_idle(bit)}
    if busy_clbits:
        raise DAGCircuitError('clbits not idle: %s' % busy_clbits)
    cregs_to_remove = {creg for creg in self.cregs.values() if not clbits.isdisjoint(creg)}
    self.remove_cregs(*cregs_to_remove)
    for clbit in clbits:
        self._remove_idle_wire(clbit)
        self.clbits.remove(clbit)
        del self._clbit_indices[clbit]
    for i, clbit in enumerate(self.clbits):
        self._clbit_indices[clbit] = self._clbit_indices[clbit]._replace(index=i)