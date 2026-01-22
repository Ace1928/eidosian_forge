from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_duplicateAdapterForClass(self):
    """
        Test that attempting to register a second adapter from a class
        raises the appropriate exception.
        """

    class TheOriginal:
        pass
    return self._duplicateAdapterForClassOrInterface(TheOriginal)