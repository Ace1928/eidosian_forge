import unittest
from traits.testing.api import UnittestTools
from traits.util.api import deprecated
def test_deprecated_exception_raising_function(self):
    with self.assertRaises(ZeroDivisionError):
        with self.assertDeprecated():
            my_bad_function()