from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_technicallyTextIsIterableBut(self) -> None:
    """
        Technically, L{str} (or L{unicode}, as appropriate) is iterable, but
        C{URL(path="foo")} resulting in C{URL.fromText("f/o/o")} is never what
        you want.
        """
    with self.assertRaises(TypeError) as raised:
        URL(path='foo')
    self.assertEqual(str(raised.exception), 'expected iterable of text for path, not: {}'.format(repr('foo')))