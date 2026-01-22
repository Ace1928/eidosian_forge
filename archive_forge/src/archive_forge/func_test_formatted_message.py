import pyomo.common.unittest as unittest
from pyomo.common.errors import format_exception
def test_formatted_message(self):
    self.assertEqual(format_exception('Hello\nworld'), 'Hello\nworld')