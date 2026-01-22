from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_clickCollapse(self) -> None:
    """
        L{URL.click} collapses C{.} and C{..} according to RFC 3986 section
        5.2.4.
        """
    tests = [['http://localhost/', '.', 'http://localhost/'], ['http://localhost/', '..', 'http://localhost/'], ['http://localhost/a/b/c', '.', 'http://localhost/a/b/'], ['http://localhost/a/b/c', '..', 'http://localhost/a/'], ['http://localhost/a/b/c', './d/e', 'http://localhost/a/b/d/e'], ['http://localhost/a/b/c', '../d/e', 'http://localhost/a/d/e'], ['http://localhost/a/b/c', '/./d/e', 'http://localhost/d/e'], ['http://localhost/a/b/c', '/../d/e', 'http://localhost/d/e'], ['http://localhost/a/b/c/', '../../d/e/', 'http://localhost/a/d/e/'], ['http://localhost/a/./c', '../d/e', 'http://localhost/d/e'], ['http://localhost/a/./c/', '../d/e', 'http://localhost/a/d/e'], ['http://localhost/a/b/c/d', './e/../f/../g', 'http://localhost/a/b/c/g'], ['http://localhost/a/b/c', 'd//e', 'http://localhost/a/b/d//e']]
    for start, click, expected in tests:
        actual = URL.fromText(start).click(click).asText()
        self.assertEqual(actual, expected, '{start}.click({click}) => {actual} not {expected}'.format(start=start, click=repr(click), actual=actual, expected=expected))