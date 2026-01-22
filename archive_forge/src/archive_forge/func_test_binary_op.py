import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_binary_op(self):

    def check_bin_op(left, op, right, result):
        code = Bytecode([Instr('LOAD_CONST', left), Instr('LOAD_CONST', right), Instr(op), Instr('STORE_NAME', 'x')])
        self.check(code, Instr('LOAD_CONST', result), Instr('STORE_NAME', 'x'))
    check_bin_op(10, 'BINARY_ADD', 20, 30)
    check_bin_op(5, 'BINARY_SUBTRACT', 1, 4)
    check_bin_op(5, 'BINARY_MULTIPLY', 3, 15)
    check_bin_op(10, 'BINARY_TRUE_DIVIDE', 3, 10 / 3)
    check_bin_op(10, 'BINARY_FLOOR_DIVIDE', 3, 3)
    check_bin_op(10, 'BINARY_MODULO', 3, 1)
    check_bin_op(2, 'BINARY_POWER', 8, 256)
    check_bin_op(1, 'BINARY_LSHIFT', 3, 8)
    check_bin_op(16, 'BINARY_RSHIFT', 3, 2)
    check_bin_op(10, 'BINARY_AND', 3, 2)
    check_bin_op(2, 'BINARY_OR', 3, 3)
    check_bin_op(2, 'BINARY_XOR', 3, 1)