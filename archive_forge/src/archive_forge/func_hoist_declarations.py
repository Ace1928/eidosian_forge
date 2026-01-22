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
def hoist_declarations(self, instructions):
    """Walks the definitions in gates/instructions to make a list of gates to declare."""
    for instruction in instructions:
        if isinstance(instruction.operation, ControlFlowOp):
            for block in instruction.operation.blocks:
                self.hoist_declarations(block.data)
            continue
        if instruction.operation in self.global_namespace or isinstance(instruction.operation, self.builtins):
            continue
        if isinstance(instruction.operation, standard_gates.CXGate):
            if instruction.operation not in self.global_namespace:
                self._register_gate(instruction.operation)
        if instruction.operation.definition is None:
            self._register_opaque(instruction.operation)
        elif not isinstance(instruction.operation, Gate):
            raise QASM3ExporterError('Exporting non-unitary instructions is not yet supported.')
        else:
            self.hoist_declarations(instruction.operation.definition.data)
            self._register_gate(instruction.operation)