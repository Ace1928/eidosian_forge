from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_proxyName(self):
    """
        The name of a proxy class indicates which interface it proxies.
        """
    proxy = proxyForInterface(IProxiedInterface)
    self.assertEqual(proxy.__name__, '(Proxy for twisted.python.test.test_components.IProxiedInterface)')