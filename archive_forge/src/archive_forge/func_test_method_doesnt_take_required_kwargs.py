import unittest
def test_method_doesnt_take_required_kwargs(self):
    from zope.interface import Interface
    from zope.interface import implementer
    from zope.interface.exceptions import BrokenMethodImplementation

    class ICurrent(Interface):

        def method(**kwargs):
            """docstring"""

    @implementer(ICurrent)
    class Current:

        def method(self, a):
            raise NotImplementedError()
    self.assertRaises(BrokenMethodImplementation, self._callFUT, ICurrent, Current)