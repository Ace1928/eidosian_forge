import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def test_no_modify_handler_not_listenable(self):
    f = Foo()
    b = BazNoModify(foo=f)
    self.assertEqual(f.u, b.u)
    f.u = 'changed'
    self.assertEqual(f.u, b.u)
    self.assertEqual(baz_u_handler_self, None)