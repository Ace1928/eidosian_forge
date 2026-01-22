from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_proxyMethod(self):
    """
        The class created from L{proxyForInterface} passes methods on an
        interface to the object which is passed to its constructor.
        """
    klass = proxyForInterface(IProxiedInterface)
    yayable = Yayable()
    proxy = klass(yayable)
    proxy.yay()
    self.assertEqual(proxy.yay(), 2)
    self.assertEqual(yayable.yays, 2)