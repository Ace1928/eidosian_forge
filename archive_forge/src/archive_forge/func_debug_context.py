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
def debug_context(self, p):
    debug_start = max(self.qasm.rfind('\n', 0, p.lexpos) + 1, p.lexpos - 5)
    debug_end = min(self.qasm.find('\n', p.lexpos, p.lexpos + 5), p.lexpos + 5)
    return '...' + self.qasm[debug_start:debug_end] + '\n' + ' ' * (3 + p.lexpos - debug_start) + '^'