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
def center_label(self, input_length, order):
    """In multi-bit elements, the label is centered vertically.

        Args:
            input_length (int): Rhe amount of wires affected.
            order (int): Which middle element is this one?
        """
    if input_length == order == 0:
        self.top_connect = self.label
        return
    location_in_the_box = '*'.center(input_length * 2 - 1).index('*') + 1
    top_limit = order * 2 + 2
    bot_limit = top_limit + 2
    if top_limit <= location_in_the_box < bot_limit:
        if location_in_the_box == top_limit:
            self.top_connect = self.label
        elif location_in_the_box == top_limit + 1:
            self.mid_content = self.label
        else:
            self.bot_connect = self.label