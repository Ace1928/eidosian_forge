from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_binary_dtype_constr(self):
    sym = datashape.TypeSymbolTable(bare=True)
    sym.dtype['int8'] = ct.int8
    sym.dtype['uint16'] = ct.uint16
    sym.dtype['float64'] = ct.float64
    sym.dtype_constr['typevar'] = ct.TypeVar
    expected_arg = [None, None]

    def _binary_type_constr(a, b):
        self.assertEqual(a, expected_arg[0])
        self.assertEqual(b, expected_arg[1])
        expected_arg[0] = None
        expected_arg[1] = None
        return ct.float32
    sym.dtype_constr['binary'] = _binary_type_constr

    def assertExpectedParse(ds_str, expected_a, expected_b):
        expected_arg[0] = expected_a
        expected_arg[1] = expected_b
        self.assertEqual(parse(ds_str, sym), ct.DataShape(ct.float32))
        self.assertEqual(expected_arg, [None, None], 'The test binary type constructor did not run')
    assertExpectedParse('binary[1, 0]', 1, 0)
    assertExpectedParse('binary[0, "test"]', 0, 'test')
    assertExpectedParse('binary[int8, "test"]', ct.DataShape(ct.int8), 'test')
    assertExpectedParse('binary[[1,3,5], "test"]', [1, 3, 5], 'test')
    assertExpectedParse('binary[0, b=1]', 0, 1)
    assertExpectedParse('binary["test", b=A]', 'test', ct.DataShape(ct.TypeVar('A')))
    assertExpectedParse('binary[[3, 6], b=int8]', [3, 6], ct.DataShape(ct.int8))
    assertExpectedParse('binary[Arg, b=["x", "test"]]', ct.DataShape(ct.TypeVar('Arg')), ['x', 'test'])
    assertExpectedParse('binary[a=1, b=0]', 1, 0)
    assertExpectedParse('binary[a=[int8, A, uint16], b="x"]', [ct.DataShape(ct.int8), ct.DataShape(ct.TypeVar('A')), ct.DataShape(ct.uint16)], 'x')