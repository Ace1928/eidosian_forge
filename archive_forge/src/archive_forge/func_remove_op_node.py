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
def remove_op_node(self, node):
    """Remove an operation node n.

        Add edges from predecessors to successors.
        """
    if not isinstance(node, DAGOpNode):
        raise DAGCircuitError('The method remove_op_node only works on DAGOpNodes. A "%s" node type was wrongly provided.' % type(node))
    self._multi_graph.remove_node_retain_edges(node._node_id, use_outgoing=False, condition=lambda edge1, edge2: edge1 == edge2)
    self._decrement_op(node.op)