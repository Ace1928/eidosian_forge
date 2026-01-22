import unittest
from traits.testing.nose_tools import deprecated, performance, skip
def test_skip_deprecated(self):
    with self.assertWarns(DeprecationWarning) as cm:

        @skip
        def some_func():
            pass
    self.assertIn('test_nose_tools', cm.filename)