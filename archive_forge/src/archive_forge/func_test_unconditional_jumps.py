import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_unconditional_jumps(self):
    label_instr7 = Label()
    code = Bytecode([Instr('LOAD_GLOBAL', 'x', lineno=2), Instr('POP_JUMP_IF_FALSE', label_instr7, lineno=2), Instr('LOAD_GLOBAL', 'y', lineno=3), Instr('POP_JUMP_IF_FALSE', label_instr7, lineno=3), Instr('LOAD_GLOBAL', 'func', lineno=4), Instr('CALL_FUNCTION', 0, lineno=4), Instr('POP_TOP', lineno=4), label_instr7, Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)])
    label_return = Label()
    self.check(code, Instr('LOAD_GLOBAL', 'x', lineno=2), Instr('POP_JUMP_IF_FALSE', label_return, lineno=2), Instr('LOAD_GLOBAL', 'y', lineno=3), Instr('POP_JUMP_IF_FALSE', label_return, lineno=3), Instr('LOAD_GLOBAL', 'func', lineno=4), Instr('CALL_FUNCTION', 0, lineno=4), Instr('POP_TOP', lineno=4), label_return, Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4))