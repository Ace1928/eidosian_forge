import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_invariant_simple(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface import directlyProvides
    from zope.interface import invariant

    class IInvariant(Interface):
        foo = Attribute('foo')
        bar = Attribute('bar; must eval to Boolean True if foo does')
        invariant(_ifFooThenBar)

    class HasInvariant:
        pass
    has_invariant = HasInvariant()
    directlyProvides(has_invariant, IInvariant)
    self.assertEqual(IInvariant.getTaggedValue('invariants'), [_ifFooThenBar])
    self.assertEqual(IInvariant.validateInvariants(has_invariant), None)
    has_invariant.bar = 27
    self.assertEqual(IInvariant.validateInvariants(has_invariant), None)
    has_invariant.foo = 42
    self.assertEqual(IInvariant.validateInvariants(has_invariant), None)
    del has_invariant.bar
    self._errorsEqual(has_invariant, 1, ['If Foo, then Bar!'], IInvariant)