import collections
import re
import io
import itertools
import numbers
from os.path import dirname, join, abspath
from typing import Iterable, List, Sequence, Union
from qiskit.circuit import (
from qiskit.circuit.bit import Bit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import (
from qiskit.circuit.library import standard_gates
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check
from . import ast
from .experimental import ExperimentalFeatures
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter
def build_gate_call(self, instruction: CircuitInstruction):
    """Builds a QuantumGateCall"""
    if isinstance(instruction.operation, standard_gates.UGate):
        gate_name = ast.Identifier('U')
    else:
        gate_name = ast.Identifier(self.global_namespace[instruction.operation])
    qubits = [self._lookup_variable(qubit) for qubit in instruction.qubits]
    if self.disable_constants:
        parameters = [ast.StringifyAndPray(self._rebind_scoped_parameters(param)) for param in instruction.operation.params]
    else:
        parameters = [ast.StringifyAndPray(pi_check(self._rebind_scoped_parameters(param), output='qasm')) for param in instruction.operation.params]
    return ast.QuantumGateCall(gate_name, qubits, parameters=parameters)