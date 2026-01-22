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
def p_format(self, p):
    """format : FORMAT_SPEC"""
    if p[1] != '2.0':
        raise QasmException(f'Unsupported OpenQASM version: {p[1]}, only 2.0 is supported currently by Cirq')