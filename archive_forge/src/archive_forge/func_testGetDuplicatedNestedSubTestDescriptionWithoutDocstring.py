import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testGetDuplicatedNestedSubTestDescriptionWithoutDocstring(self):
    with self.subTest(foo=1, bar=2):
        with self.subTest(baz=3, bar=4):
            result = unittest.TextTestResult(None, True, 1)
            self.assertEqual(result.getDescription(self._subtest), 'testGetDuplicatedNestedSubTestDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetDuplicatedNestedSubTestDescriptionWithoutDocstring) (baz=3, bar=4, foo=1)')