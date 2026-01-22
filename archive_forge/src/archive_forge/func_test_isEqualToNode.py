from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_isEqualToNode(self) -> None:
    """
        L{Element.isEqualToNode} returns C{True} if and only if passed a
        L{Element} with the same C{nodeName}, C{namespace}, C{childNodes}, and
        C{attributes}.
        """
    self.assertTrue(microdom.Element('foo', {'a': 'b'}, object(), namespace='bar').isEqualToNode(microdom.Element('foo', {'a': 'b'}, object(), namespace='bar')))
    self.assertFalse(microdom.Element('foo', {'a': 'b'}, object(), namespace='bar').isEqualToNode(microdom.Element('bar', {'a': 'b'}, object(), namespace='bar')))
    self.assertFalse(microdom.Element('foo', {'a': 'b'}, object(), namespace='bar').isEqualToNode(microdom.Element('foo', {'a': 'b'}, object(), namespace='baz')))
    one = microdom.Element('foo', {'a': 'b'}, object(), namespace='bar')
    two = microdom.Element('foo', {'a': 'b'}, object(), namespace='bar')
    two.appendChild(microdom.Node(object()))
    self.assertFalse(one.isEqualToNode(two))
    self.assertFalse(microdom.Element('foo', {'a': 'b'}, object(), namespace='bar').isEqualToNode(microdom.Element('foo', {'a': 'c'}, object(), namespace='bar')))