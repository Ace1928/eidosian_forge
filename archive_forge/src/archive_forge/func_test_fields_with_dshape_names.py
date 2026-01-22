from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_fields_with_dshape_names(self):
    ds = parse('{\n                type: bool,\n                data: bool,\n                blob: bool,\n                bool: bool,\n                int: int32,\n                float: float32,\n                double: float64,\n                int8: int8,\n                int16: int16,\n                int32: int32,\n                int64: int64,\n                uint8: uint8,\n                uint16: uint16,\n                uint32: uint32,\n                uint64: uint64,\n                float16: float32,\n                float32: float32,\n                float64: float64,\n                float128: float64,\n                complex: float32,\n                complex64: float32,\n                complex128: float64,\n                string: string,\n                object: string,\n                datetime: string,\n                datetime64: string,\n                timedelta: string,\n                timedelta64: string,\n                json: string,\n                var: string,\n            }', self.sym)
    self.assertEqual(type(ds[-1]), ct.Record)
    self.assertEqual(len(ds[-1].names), 30)