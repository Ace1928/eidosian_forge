from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_subclassAdapterRegistrationForInterface(self):
    """
        Test that an adapter to a particular interface can be registered
        from both an interface and its subclass.
        """
    return self._subclassAdapterRegistrationForClassOrInterface(ITest2)