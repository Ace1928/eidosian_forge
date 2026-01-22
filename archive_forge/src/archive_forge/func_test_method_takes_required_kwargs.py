import unittest
def test_method_takes_required_kwargs(self):
    from zope.interface import Interface
    from zope.interface import implementer

    class ICurrent(Interface):

        def method(**kwargs):
            """docstring"""

    @implementer(ICurrent)
    class Current:

        def method(self, **kw):
            raise NotImplementedError()
    self._callFUT(ICurrent, Current)