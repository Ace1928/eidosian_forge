from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_registerAdapterForClass(self):
    """
        Test that an adapter from a class can be registered and then looked
        up.
        """

    class TheOriginal:
        pass
    return self._registerAdapterForClassOrInterface(TheOriginal)