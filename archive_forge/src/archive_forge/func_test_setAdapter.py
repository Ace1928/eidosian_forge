from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_setAdapter(self):
    """
        C{Componentized.setAdapter} sets a component for an interface by
        wrapping the instance with the given adapter class.
        """
    componentized = components.Componentized()
    componentized.setAdapter(IAdept, Adept)
    component = componentized.getComponent(IAdept)
    self.assertEqual(component.original, componentized)
    self.assertIsInstance(component, Adept)