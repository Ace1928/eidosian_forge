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
def build_for_loop(self, instruction: CircuitInstruction) -> ast.ForLoopStatement:
    """Build a :obj:`.ForLoopOp` into a :obj:`.ast.ForLoopStatement`."""
    indexset, loop_parameter, loop_circuit = instruction.operation.params
    self.push_scope(loop_circuit, instruction.qubits, instruction.clbits)
    scope = self.current_scope()
    if loop_parameter is None:
        loop_parameter_ast = self._reserve_variable_name(ast.Identifier('_'), scope)
    else:
        loop_parameter_ast = self._register_variable(loop_parameter, scope)
    if isinstance(indexset, range):
        indexset_ast = ast.Range(start=self.build_integer(indexset.start), end=self.build_integer(indexset.stop - 1), step=self.build_integer(indexset.step) if indexset.step != 1 else None)
    else:
        try:
            indexset_ast = ast.IndexSet([self.build_integer(value) for value in indexset])
        except QASM3ExporterError:
            raise QASM3ExporterError(f"The values in OpenQASM 3 'for' loops must all be integers, but received '{indexset}'.") from None
    body_ast = self.build_program_block(loop_circuit)
    self.pop_scope()
    return ast.ForLoopStatement(indexset_ast, loop_parameter_ast, body_ast)