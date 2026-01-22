import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import get_code, TestCase
def test_extreme_compute_jumps_convergence(self):
    """Test of compute_jumps() requiring absurd number of passes.

        NOTE:  This test also serves to demonstrate that there is no worst
        case: the number of passes can be unlimited (or, actually, limited by
        the size of the provided code).

        This is an extension of test_compute_jumps_convergence.  Instead of
        two jumps, where the earlier gets extended after the latter, we
        instead generate a series of many jumps.  Each pass of compute_jumps()
        extends one more instruction, which in turn causes the one behind it
        to be extended on the next pass.

        """
    max_unextended_offset = 1 << 8
    unextended_branch_instr_size = 2
    N = max_unextended_offset // unextended_branch_instr_size
    if OFFSET_AS_INSTRUCTION:
        N *= 2
    nop = 'UNARY_POSITIVE'
    labels = [Label() for x in range(0, 3 * N)]
    code = Bytecode()
    code.extend((Instr('JUMP_FORWARD', labels[len(labels) - x - 1]) for x in range(0, len(labels))))
    end_of_jumps = len(code)
    code.extend((Instr(nop) for x in range(0, N)))
    offset = end_of_jumps + N
    for index in range(0, len(labels)):
        code.insert(offset, labels[index])
        if offset <= end_of_jumps:
            offset -= 1
        else:
            offset -= 2
    code.insert(0, Instr('LOAD_CONST', 0))
    del end_of_jumps
    code.append(Instr('RETURN_VALUE'))
    code.to_code(compute_jumps_passes=len(labels) + 1)