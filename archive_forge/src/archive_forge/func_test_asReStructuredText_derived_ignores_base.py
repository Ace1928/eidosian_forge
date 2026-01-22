import sys
import unittest
def test_asReStructuredText_derived_ignores_base(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    EXPECTED = '\n\n'.join(['``IDerived``', ' IDerived doc', ' This interface extends:', '  o ``IBase``', ' Attributes:', '  ``attr1`` -- no documentation', '  ``attr2`` -- attr2 doc', ' Methods:', '  ``method3()`` -- method3 doc', '  ``method4()`` -- no documentation', '  ``method5()`` -- method5 doc', ''])

    class IBase(Interface):

        def method1():
            pass

        def method2():
            pass

    class IDerived(IBase):
        """IDerived doc"""
        attr1 = Attribute('attr1')
        attr2 = Attribute('attr2', 'attr2 doc')

        def method3():
            """method3 doc"""

        def method4():
            pass

        def method5():
            """method5 doc"""
    self.assertEqual(self._callFUT(IDerived), EXPECTED)