from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_new_type_str(self):
    self.assertEqual(tostr(DerivedStr(1)), '1')
    self.assertIs(tostr.handlers[DerivedStr], tostr.handlers[str])