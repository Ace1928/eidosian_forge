import pyomo.common.unittest as unittest
from pyomo.common.errors import format_exception
def test_long_basic_message(self):
    self.assertEqual(format_exception('Hello world, this is a very long message that will inevitably wrap onto another line.'), 'Hello world, this is a very long message that will\n    inevitably wrap onto another line.')