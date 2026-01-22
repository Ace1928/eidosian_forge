from io import StringIO
from warnings import warn
from shutil import get_terminal_size
import collections
import sys
from qiskit.circuit import Qubit, Clbit, ClassicalRegister
from qiskit.circuit import ControlledGate, Reset, Measure
from qiskit.circuit import ControlFlowOp, WhileLoopOp, IfElseOp, ForLoopOp, SwitchCaseOp
from qiskit.circuit.classical import expr
from qiskit.circuit.controlflow import node_resources
from qiskit.circuit.library.standard_gates import IGate, RZZGate, SwapGate, SXGate, SXdgGate
from qiskit.circuit.annotated_operation import _canonicalize_modifiers, ControlModifier
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter
from ._utils import (
from ..exceptions import VisualizationError
@staticmethod
def controlled_wires(node, wire_map, ctrl_text, conditional, mod_control):
    """
        Analyzes the node in the layer and checks if the controlled arguments are in
        the box or out of the box.

        Args:
            node (DAGNode): node to analyse
            wire_map (dict): map of qubits/clbits to position
            ctrl_text (str): text for a control label
            conditional (bool): is this a node with a condition
            mod_control (ControlModifier): an instance of a modifier for an
                AnnotatedOperation

        Returns:
            Tuple(list, list, list):
              - tuple: controlled arguments on top of the "node box", and its status
              - tuple: controlled arguments on bottom of the "node box", and its status
              - tuple: controlled arguments in the "node box", and its status
              - the rest of the arguments
        """
    op = node.op
    num_ctrl_qubits = mod_control.num_ctrl_qubits if mod_control else op.num_ctrl_qubits
    ctrl_qubits = node.qargs[:num_ctrl_qubits]
    args_qubits = node.qargs[num_ctrl_qubits:]
    ctrl_state = mod_control.ctrl_state if mod_control else op.ctrl_state
    ctrl_state = f'{ctrl_state:b}'.rjust(num_ctrl_qubits, '0')[::-1]
    in_box = []
    top_box = []
    bot_box = []
    qubit_indices = sorted((wire_map[x] for x in wire_map if x in args_qubits))
    for ctrl_qubit in zip(ctrl_qubits, ctrl_state):
        if min(qubit_indices) > wire_map[ctrl_qubit[0]]:
            top_box.append(ctrl_qubit)
        elif max(qubit_indices) < wire_map[ctrl_qubit[0]]:
            bot_box.append(ctrl_qubit)
        else:
            in_box.append(ctrl_qubit)
    gates = []
    for i in range(len(ctrl_qubits)):
        if getattr(op, 'condition', None) is not None:
            conditional = True
        if ctrl_state[i] == '1':
            gates.append(Bullet(conditional=conditional, label=ctrl_text, bottom=bool(bot_box)))
        else:
            gates.append(OpenBullet(conditional=conditional, label=ctrl_text, bottom=bool(bot_box)))
    return (gates, top_box, bot_box, in_box, args_qubits)