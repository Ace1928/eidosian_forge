from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_implements(self):
    """
        The resulting proxy implements the interface that it proxies.
        """
    proxy = proxyForInterface(IProxiedInterface)
    self.assertTrue(IProxiedInterface.implementedBy(proxy))