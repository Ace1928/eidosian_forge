from typing import Any
from unittest import TestCase
from .common import HyperlinkTestCase
def test_assertRaisesContextManager(self):
    """HyperlinkTestCase.assertRaises does not raise an AssertionError
        when used as a context manager with a suite that raises the
        expected exception.  The context manager stores the exception
        instance under its `exception` instance variable.

        """
    with self.hyperlink_test.assertRaises(_ExpectedException) as cm:
        raise _ExpectedException
    self.assertTrue(isinstance(cm.exception, _ExpectedException))