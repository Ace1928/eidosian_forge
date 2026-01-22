from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_dtype_constr_errors(self):
    sym = datashape.TypeSymbolTable(bare=True)
    sym.dtype['int8'] = ct.int8
    sym.dtype['uint16'] = ct.uint16
    sym.dtype['float64'] = ct.float64

    def _type_constr(*args, **kwargs):
        return ct.float32
    sym.dtype_constr['tcon'] = _type_constr
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[]', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[unknown]', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[x=', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[x=]', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[x=A, B]', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[[0, "x"]]', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[[0, X]]', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[["x", 0]]', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[["x", X]]', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[[X, 0]]', sym)
    self.assertRaises(DataShapeSyntaxError, parse, 'tcon[[X, "x"]]', sym)