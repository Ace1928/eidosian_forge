from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_queryIterable(self) -> None:
    """
        When a L{URL} is created with a C{query} argument, the C{query}
        argument is converted into an N-tuple of 2-tuples.
        """
    url = URL(query=[['alpha', 'beta']])
    self.assertEqual(url.query, (('alpha', 'beta'),))