from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_mailto(self) -> None:
    """
        Although L{URL} instances are mainly for dealing with HTTP, other
        schemes (such as C{mailto:}) should work as well.  For example,
        L{URL.fromText}/L{URL.asText} round-trips cleanly for a C{mailto:} URL
        representing an email address.
        """
    self.assertEqual(URL.fromText('mailto:user@example.com').asText(), 'mailto:user@example.com')