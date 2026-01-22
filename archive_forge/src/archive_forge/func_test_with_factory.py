import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Any
def test_with_factory(self):

    class A(HasTraits):
        foo = Any(factory=dict)
    a = A()
    b = A()
    self.assertEqual(a.foo, {})
    self.assertEqual(b.foo, {})
    a.foo['key'] = 23
    self.assertEqual(a.foo, {'key': 23})
    self.assertEqual(b.foo, {})
    a.foo = b.foo = {'red': 16711680}
    a.foo['green'] = 65280
    self.assertEqual(b.foo['green'], 65280)