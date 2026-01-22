from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_duplicateAdapterForInterfaceAllowed(self):
    """
        Test that when L{components.ALLOW_DUPLICATES} is set to a true
        value, duplicate registrations from interfaces are allowed to
        override the original registration.
        """

    class TheOriginal(Interface):
        pass
    return self._duplicateAdapterForClassOrInterfaceAllowed(TheOriginal)