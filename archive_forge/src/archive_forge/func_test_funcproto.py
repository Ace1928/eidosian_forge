from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_funcproto(sym):
    assert parse('(float32) -> float64', sym) == ct.DataShape(ct.Function(ct.DataShape(ct.float32), ct.DataShape(ct.float64)))
    assert parse('(int16, int32) -> bool', sym) == ct.DataShape(ct.Function(ct.DataShape(ct.int16), ct.DataShape(ct.int32), ct.DataShape(ct.bool_)))
    assert parse('(float32,) -> float64', sym) == ct.DataShape(ct.Function(ct.DataShape(ct.float32), ct.DataShape(ct.float64)))
    assert_dshape_equal(parse('(int16, int32,) -> bool', sym), ct.DataShape(ct.Function(ct.DataShape(ct.int16), ct.DataShape(ct.int32), ct.DataShape(ct.bool_))))
    assert_dshape_equal(parse('() -> bool', sym), ct.DataShape(ct.Function(ct.DataShape(ct.bool_))))