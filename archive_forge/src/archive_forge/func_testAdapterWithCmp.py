from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def testAdapterWithCmp(self):
    components.registerAdapter(DoubleXAdapter, IAttrX, IAttrXX)
    xx = IAttrXX(Xcellent())
    self.assertEqual(('x!', 'x!'), xx.xx())