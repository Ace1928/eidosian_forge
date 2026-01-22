from twisted.python import reflect, win32
from twisted.trial import unittest
def test_emptyStringArg(self) -> None:
    """
        Calling C{cmdLineQuote} with an empty string returns a quoted empty
        string.
        """
    self.assertEqual(win32.cmdLineQuote(''), '""')