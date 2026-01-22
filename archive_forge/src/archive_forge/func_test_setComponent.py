from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_setComponent(self):
    """
        C{Componentized.setComponent} stores the given component using the
        given interface as the key.
        """
    componentized = components.Componentized()
    obj = object()
    componentized.setComponent(ITest, obj)
    self.assertIs(componentized.getComponent(ITest), obj)