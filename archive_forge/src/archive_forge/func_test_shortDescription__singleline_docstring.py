import unittest
from unittest.test.support import LoggingResult
def test_shortDescription__singleline_docstring(self):
    desc = 'this tests foo'
    test = unittest.FunctionTestCase(lambda: None, description=desc)
    self.assertEqual(test.shortDescription(), 'this tests foo')