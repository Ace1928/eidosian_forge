import unittest
def test_method_doesnt_take_required_starargs(self):
    from zope.interface import Interface
    from zope.interface import implementer
    from zope.interface.exceptions import BrokenMethodImplementation

    class ICurrent(Interface):

        def method(*args):
            """docstring"""

    @implementer(ICurrent)
    class Current:

        def method(self):
            raise NotImplementedError()
    self.assertRaises(BrokenMethodImplementation, self._callFUT, ICurrent, Current)