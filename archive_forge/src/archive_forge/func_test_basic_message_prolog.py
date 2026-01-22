import pyomo.common.unittest as unittest
from pyomo.common.errors import format_exception
def test_basic_message_prolog(self):
    self.assertEqual(format_exception('This is a very, very, very long message that will inevitably wrap onto another line.', prolog='Hello world:'), 'Hello world:\n    This is a very, very, very long message that will inevitably wrap onto\n    another line.')