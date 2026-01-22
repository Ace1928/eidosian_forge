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
def build_while_loop(self, instruction: CircuitInstruction) -> ast.WhileLoopStatement:
    """Build a :obj:`.WhileLoopOp` into a :obj:`.ast.WhileLoopStatement`."""
    condition = self.build_expression(_lift_condition(instruction.operation.condition))
    loop_circuit = instruction.operation.blocks[0]
    self.push_scope(loop_circuit, instruction.qubits, instruction.clbits)
    loop_body = self.build_program_block(loop_circuit.data)
    self.pop_scope()
    return ast.WhileLoopStatement(condition, loop_body)