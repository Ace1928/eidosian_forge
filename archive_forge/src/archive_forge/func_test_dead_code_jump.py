import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_dead_code_jump(self):
    label = Label()
    code = Bytecode([Instr('LOAD_NAME', 'x'), Instr('JUMP_ABSOLUTE', label), Instr('LOAD_NAME', 'y'), Instr('STORE_NAME', 'test'), label, Instr('STORE_NAME', 'test')])
    self.check(code, Instr('LOAD_NAME', 'x'), Instr('STORE_NAME', 'test'))