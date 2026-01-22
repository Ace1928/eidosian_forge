import pyomo.common.unittest as unittest
from pyomo.common.errors import format_exception
def test_basic_message_long_prolog(self):
    msg = format_exception('This is a very, very, very long message that will inevitably wrap onto another line.', prolog='Hello, this is a more verbose prolog that will trigger a line wrap:')
    self.assertEqual(msg, 'Hello, this is a more verbose prolog that will trigger\n    a line wrap:\n        This is a very, very, very long message that will inevitably wrap\n        onto another line.')