from __future__ import annotations
from twisted.internet.abstract import isIPv6Address
from twisted.trial.unittest import SynchronousTestCase
def test_colon(self) -> None:
    """
        A single C{":"} is not an IPv6 address literal.
        """
    self.assertFalse(isIPv6Address(':'))