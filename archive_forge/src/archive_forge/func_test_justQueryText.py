from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_justQueryText(self) -> None:
    """
        An L{URL} with query text should serialize as just query text.
        """
    u = URL(query=[('hello', 'world')])
    self.assertEqual(u.asText(), '?hello=world')