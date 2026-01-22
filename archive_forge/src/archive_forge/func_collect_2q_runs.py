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
def collect_2q_runs(self):
    """Return a set of non-conditional runs of 2q "op" nodes."""
    to_qid = {}
    for i, qubit in enumerate(self.qubits):
        to_qid[qubit] = i

    def filter_fn(node):
        if isinstance(node, DAGOpNode):
            return isinstance(node.op, Gate) and len(node.qargs) <= 2 and (not getattr(node.op, 'condition', None)) and (not node.op.is_parameterized())
        else:
            return None

    def color_fn(edge):
        if isinstance(edge, Qubit):
            return to_qid[edge]
        else:
            return None
    return rx.collect_bicolor_runs(self._multi_graph, filter_fn, color_fn)