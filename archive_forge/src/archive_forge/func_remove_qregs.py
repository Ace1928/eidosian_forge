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
def remove_qregs(self, *qregs):
    """
        Remove classical registers from the circuit, leaving underlying bits
        in place.

        Raises:
            DAGCircuitError: a qreg is not a QuantumRegister, or is not in
            the circuit.
        """
    if any((not isinstance(qreg, QuantumRegister) for qreg in qregs)):
        raise DAGCircuitError('qregs not of type QuantumRegister: %s' % [r for r in qregs if not isinstance(r, QuantumRegister)])
    unknown_qregs = set(qregs).difference(self.qregs.values())
    if unknown_qregs:
        raise DAGCircuitError('qregs not in circuit: %s' % unknown_qregs)
    for qreg in qregs:
        del self.qregs[qreg.name]
        for j in range(qreg.size):
            bit = qreg[j]
            bit_position = self._qubit_indices[bit]
            bit_position.registers.remove((qreg, j))