from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_original(self):
    """
        Proxy objects should have an C{original} attribute which refers to the
        original object passed to the constructor.
        """
    original = object()
    proxy = proxyForInterface(IProxiedInterface)(original)
    self.assertIs(proxy.original, original)