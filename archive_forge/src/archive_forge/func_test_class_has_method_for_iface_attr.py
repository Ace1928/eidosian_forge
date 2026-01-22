import unittest
def test_class_has_method_for_iface_attr(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface import implementer

    class ICurrent(Interface):
        attr = Attribute('The foo Attribute')

    @implementer(ICurrent)
    class Current:

        def attr(self):
            raise NotImplementedError()
    self._callFUT(ICurrent, Current)