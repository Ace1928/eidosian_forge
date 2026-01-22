from twisted.python.compat import networkString
from twisted.trial import unittest
from twisted.words.protocols.jabber import sasl_mechanisms
def test_getInitialResponse(self) -> None:
    """
        Test that no initial response is generated.
        """
    self.assertIdentical(self.mechanism.getInitialResponse(), None)