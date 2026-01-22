from twisted.conch.insults import text
from twisted.conch.insults.text import attributes as A
from twisted.trial import unittest
def test_reverseVideo(self) -> None:
    """
        The reverse-video formatting attribute, L{A.reverseVideo}, emits the
        VT102 control sequence to enable reversed video when flattened.
        """
    self.assertEqual(text.assembleFormattedText(A.reverseVideo['Hello, world.']), '\x1b[7mHello, world.')