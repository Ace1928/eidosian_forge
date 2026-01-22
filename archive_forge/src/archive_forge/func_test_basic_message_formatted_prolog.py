import pyomo.common.unittest as unittest
from pyomo.common.errors import format_exception
def test_basic_message_formatted_prolog(self):
    msg = format_exception('This is a very, very, very long message that will inevitably wrap onto another line.', prolog='Hello world:\n    This is a prolog:')
    self.assertEqual(msg, 'Hello world:\n    This is a prolog:\n        This is a very, very, very long message that will inevitably wrap\n        onto another line.')