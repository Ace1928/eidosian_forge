from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def testInheritanceAdaptation(self):
    c = CComp()
    co1 = c.getComponent(ITest)
    co2 = c.getComponent(ITest)
    co3 = c.getComponent(ITest2)
    co4 = c.getComponent(ITest2)
    assert co1 is co2
    assert co3 is not co4
    c.removeComponent(co1)
    co5 = c.getComponent(ITest)
    co6 = c.getComponent(ITest)
    assert co5 is co6
    assert co1 is not co5