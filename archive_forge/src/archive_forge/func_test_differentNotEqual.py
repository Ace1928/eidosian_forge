from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_differentNotEqual(self) -> None:
    """
        L{URL}s that refer to different resources are both unequal (C{!=}) and
        also not equal (not C{==}).
        """
    u1 = URL.fromText('http://localhost/a')
    u2 = URL.fromText('http://localhost/b')
    self.assertFalse(u1 == u2, f'{u1!r} != {u2!r}')
    self.assertNotEqual(u1, u2)