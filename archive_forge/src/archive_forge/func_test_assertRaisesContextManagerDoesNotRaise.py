from typing import Any
from unittest import TestCase
from .common import HyperlinkTestCase
def test_assertRaisesContextManagerDoesNotRaise(self):
    """HyperlinkTestcase.assertRaises raises an AssertionError when used
        as a context manager with a block that does not raise any
        exception.

        """
    try:
        with self.hyperlink_test.assertRaises(_ExpectedException):
            pass
    except AssertionError:
        pass