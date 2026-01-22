from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_string_atom(self):
    self.assertEqual(dshape('string'), dshape("string['U8']"))
    self.assertEqual(dshape("string['ascii']")[0].encoding, 'A')
    self.assertEqual(dshape("string['A']")[0].encoding, 'A')
    self.assertEqual(dshape("string['utf-8']")[0].encoding, 'U8')
    self.assertEqual(dshape("string['U8']")[0].encoding, 'U8')
    self.assertEqual(dshape("string['utf-16']")[0].encoding, 'U16')
    self.assertEqual(dshape("string['U16']")[0].encoding, 'U16')
    self.assertEqual(dshape("string['utf-32']")[0].encoding, 'U32')
    self.assertEqual(dshape("string['U32']")[0].encoding, 'U32')