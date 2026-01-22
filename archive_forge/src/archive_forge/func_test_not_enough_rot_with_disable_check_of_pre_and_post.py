import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_not_enough_rot_with_disable_check_of_pre_and_post(self):
    opnames = ['ROT_TWO', 'ROT_THREE']
    if sys.version_info >= (3, 8):
        opnames.append('ROT_FOUR')
    for opname in opnames:
        with self.subTest():
            code = Bytecode()
            code.first_lineno = 1
            code.extend([Instr('LOAD_CONST', 1), Instr(opname)])
            co = code.to_code(check_pre_and_post=False)
            self.assertEqual(co.co_stacksize, 1)