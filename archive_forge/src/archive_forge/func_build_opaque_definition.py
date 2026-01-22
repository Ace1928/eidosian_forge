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
def build_opaque_definition(self, instruction):
    """Builds an Opaque gate definition as a CalibrationDefinition"""
    raise QASM3ExporterError(f'Exporting opaque instructions with pulse-level calibrations is not yet supported by the OpenQASM 3 exporter. Received this instruction, which appears opaque:\n{instruction}')