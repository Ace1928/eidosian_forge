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
def p_measurement(self, p):
    """measurement : MEASURE qarg ARROW carg ';'"""
    qreg = p[2]
    creg = p[4]
    if len(qreg) != len(creg):
        raise QasmException(f'mismatched register sizes {len(qreg)} -> {len(creg)} for measurement at line {p.lineno(1)}')
    p[0] = [ops.MeasurementGate(num_qubits=1, key=creg[i]).on(qreg[i]) for i in range(len(qreg))]