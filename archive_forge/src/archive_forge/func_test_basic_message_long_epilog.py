import pyomo.common.unittest as unittest
from pyomo.common.errors import format_exception
def test_basic_message_long_epilog(self):
    self.assertEqual(format_exception('This is a very, very, very long message that will inevitably wrap onto another line.', epilog='Hello, this is a very, very, very verbose epilog that will trigger a line wrap'), 'This is a very, very, very long message that will\n        inevitably wrap onto another line.\n    Hello, this is a very, very, very verbose epilog that will trigger a\n    line wrap')