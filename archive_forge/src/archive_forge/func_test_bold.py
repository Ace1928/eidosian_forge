from twisted.conch.insults import text
from twisted.conch.insults.text import attributes as A
from twisted.trial import unittest
def test_bold(self) -> None:
    """
        The bold formatting attribute, L{A.bold}, emits the VT102 control
        sequence to enable bold when flattened.
        """
    self.assertEqual(text.assembleFormattedText(A.bold['Hello, world.']), '\x1b[1mHello, world.')