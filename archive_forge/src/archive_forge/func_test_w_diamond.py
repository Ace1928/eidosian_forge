import unittest
def test_w_diamond(self):

    class Foo:
        pass

    class Bar(Foo):
        pass

    class Baz(Foo):
        pass

    class Qux(Bar, Baz):
        pass
    self.assertEqual(self._callFUT(Qux), [Qux, Bar, Baz, Foo, object])