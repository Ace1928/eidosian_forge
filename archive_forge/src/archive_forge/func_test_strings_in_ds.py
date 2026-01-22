from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_strings_in_ds(self):
    ds = parse("5 * var * {\n              id: int64,\n             'my field': string,\n              name: string }\n             ", self.sym)
    self.assertEqual(len(ds[-1].names), 3)
    ds = parse('2 * var * {\n             "AASD @#$@#$ \' sdf": string,\n              id: float32,\n              id2: int64,\n              name: string }\n             ', self.sym)
    self.assertEqual(len(ds[-1].names), 4)