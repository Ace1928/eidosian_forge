import unittest
def test_method_takes_wrong_arg_names_but_OK(self):
    from zope.interface import Interface
    from zope.interface import implementer

    class ICurrent(Interface):

        def method(a):
            """docstring"""

    @implementer(ICurrent)
    class Current:

        def method(self, b):
            raise NotImplementedError()
    self._callFUT(ICurrent, Current)