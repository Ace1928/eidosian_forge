import functools
import operator
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Union, TYPE_CHECKING
import numpy as np
import sympy
from ply import yacc
from cirq import ops, Circuit, NamedQubit, CX
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import._lexer import QasmLexer
from cirq.contrib.qasm_import.exception import QasmException
def p_qasm_no_format_specified_error(self, p):
    """qasm : QELIBINC
        | circuit"""
    if self.supported_format is False:
        raise QasmException("Missing 'OPENQASM 2.0;' statement")