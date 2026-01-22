import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
@unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
def testGetDescriptionWithMultiLineDocstring(self):
    """Tests getDescription() for a method with a longer docstring.
        The second line of the docstring.
        """
    result = unittest.TextTestResult(None, True, 1)
    self.assertEqual(result.getDescription(self), 'testGetDescriptionWithMultiLineDocstring (' + __name__ + '.Test_TextTestResult.testGetDescriptionWithMultiLineDocstring)\nTests getDescription() for a method with a longer docstring.')