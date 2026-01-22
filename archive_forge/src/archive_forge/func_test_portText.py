from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_portText(self) -> None:
    """
        L{URL.fromText} parses custom port numbers as integers.
        """
    portURL = URL.fromText('http://www.example.com:8080/')
    self.assertEqual(portURL.port, 8080)
    self.assertEqual(portURL.asText(), 'http://www.example.com:8080/')