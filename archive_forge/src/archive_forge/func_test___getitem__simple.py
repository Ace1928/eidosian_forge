import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___getitem__simple(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface.interface import Method

    class ISimple(Interface):
        attr = Attribute('My attr')

        def method():
            """My method"""
    a_desc = ISimple['attr']
    self.assertTrue(isinstance(a_desc, Attribute))
    self.assertEqual(a_desc.__name__, 'attr')
    self.assertEqual(a_desc.__doc__, 'My attr')
    m_desc = ISimple['method']
    self.assertTrue(isinstance(m_desc, Method))
    self.assertEqual(m_desc.__name__, 'method')
    self.assertEqual(m_desc.__doc__, 'My method')