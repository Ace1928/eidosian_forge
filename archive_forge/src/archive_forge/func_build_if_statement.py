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
def build_if_statement(self, instruction: CircuitInstruction) -> ast.BranchingStatement:
    """Build an :obj:`.IfElseOp` into a :obj:`.ast.BranchingStatement`."""
    condition = self.build_expression(_lift_condition(instruction.operation.condition))
    true_circuit = instruction.operation.blocks[0]
    self.push_scope(true_circuit, instruction.qubits, instruction.clbits)
    true_body = self.build_program_block(true_circuit.data)
    self.pop_scope()
    if len(instruction.operation.blocks) == 1:
        return ast.BranchingStatement(condition, true_body, None)
    false_circuit = instruction.operation.blocks[1]
    self.push_scope(false_circuit, instruction.qubits, instruction.clbits)
    false_body = self.build_program_block(false_circuit.data)
    self.pop_scope()
    return ast.BranchingStatement(condition, true_body, false_body)