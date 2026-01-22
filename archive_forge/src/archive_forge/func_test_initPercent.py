from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_initPercent(self) -> None:
    """
        L{URL} should accept (and not interpret) percent characters.
        """
    u = URL('s', '%68', ['%70'], [('%6B', '%76'), ('%6B', None)], '%66')
    self.assertUnicoded(u)
    self.assertURL(u, 's', '%68', ['%70'], [('%6B', '%76'), ('%6B', None)], '%66', None)