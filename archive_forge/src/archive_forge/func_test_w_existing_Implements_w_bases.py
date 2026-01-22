import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_w_existing_Implements_w_bases(self):
    from zope.interface.declarations import Implements
    from zope.interface.interface import InterfaceClass
    IRoot = InterfaceClass('IRoot')
    ISecondRoot = InterfaceClass('ISecondRoot')
    IExtendsRoot = InterfaceClass('IExtendsRoot', (IRoot,))
    impl_root = Implements.named('Root', IRoot)
    impl_root.declared = (IRoot,)

    class Root1:
        __implemented__ = impl_root

    class Root2:
        __implemented__ = impl_root
    impl_extends_root = Implements.named('ExtendsRoot1', IExtendsRoot)
    impl_extends_root.declared = (IExtendsRoot,)

    class ExtendsRoot(Root1, Root2):
        __implemented__ = impl_extends_root
    impl_extends_root.inherit = ExtendsRoot
    self._callFUT(ExtendsRoot, ISecondRoot)
    self.assertIs(ExtendsRoot.__implemented__, impl_extends_root)
    self.assertEqual(impl_extends_root.inherit, ExtendsRoot)
    self.assertEqual(impl_extends_root.declared, self._order_for_two(IExtendsRoot, ISecondRoot))
    self.assertEqual(impl_extends_root.__bases__, self._order_for_two(IExtendsRoot, ISecondRoot) + (impl_root,))