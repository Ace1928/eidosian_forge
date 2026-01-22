import re
from typing import Optional
import numpy as np
import ply.lex as lex
from cirq.contrib.qasm_import.exception import QasmException
def t_QELIBINC(self, t):
    """include(\\s+)"qelib1.inc";"""
    return t