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
def set_clbit(self, clbit, element):
    """Sets the clbit to the element.

        Args:
            clbit (cbit): Element of self.clbits.
            element (DrawElement): Element to set in the clbit
        """
    register = get_bit_register(self._circuit, clbit)
    if self.cregbundle and register is not None:
        self.clbit_layer[self._wire_map[register] - len(self.qubits)] = element
    else:
        self.clbit_layer[self._wire_map[clbit] - len(self.qubits)] = element