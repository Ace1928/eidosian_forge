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
def p_expr_function_call(self, p):
    """expr : ID '(' expr ')'"""
    func = p[1]
    if func not in self.functions.keys():
        raise QasmException(f"Function not recognized: '{func}' at line {p.lineno(1)}")
    p[0] = self.functions[func](p[3])