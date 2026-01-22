from twisted.conch.insults import text
from twisted.conch.insults.text import attributes as A
from twisted.trial import unittest
def test_foreground(self) -> None:
    """
        The foreground color formatting attribute, L{A.fg}, emits the VT102
        control sequence to set the selected foreground color when flattened.
        """
    self.assertEqual(text.assembleFormattedText(A.normal[A.fg.red['Hello, '], A.fg.green['world!']]), '\x1b[31mHello, \x1b[32mworld!')