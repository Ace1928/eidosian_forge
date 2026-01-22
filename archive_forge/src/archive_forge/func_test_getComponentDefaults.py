from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_getComponentDefaults(self):
    """
        Test that a default value specified to Componentized.getComponent if
        there is no component for the requested interface.
        """
    componentized = components.Componentized()
    default = object()
    self.assertIs(componentized.getComponent(ITest, default), default)
    self.assertIs(componentized.getComponent(ITest, default=default), default)
    self.assertIs(componentized.getComponent(ITest), None)