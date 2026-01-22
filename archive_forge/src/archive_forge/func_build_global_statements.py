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
def build_global_statements(self) -> List[ast.Statement]:
    """Get a list of the statements that form the global scope of the program."""
    definitions = self.build_definitions()
    self.build_parameter_declarations()
    self.build_classical_declarations()
    context = self.global_scope(assert_=True).circuit
    quantum_declarations = self.build_quantum_declarations()
    quantum_instructions = self.build_quantum_instructions(context.data)
    return [statement for source in (self._global_io_declarations, definitions, self._global_classical_declarations, quantum_declarations, quantum_instructions) for statement in source]