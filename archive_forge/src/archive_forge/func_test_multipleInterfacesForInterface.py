from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_multipleInterfacesForInterface(self):
    """
        Test the registration of an adapter from an interface to several
        interfaces at once.
        """
    return self._multipleInterfacesForClassOrInterface(ITest3)