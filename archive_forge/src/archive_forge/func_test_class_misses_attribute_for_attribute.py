import unittest
def test_class_misses_attribute_for_attribute(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface import implementer
    from zope.interface.exceptions import BrokenImplementation

    class ICurrent(Interface):
        attr = Attribute('The foo Attribute')

    @implementer(ICurrent)
    class Current:
        pass
    self.assertRaises(BrokenImplementation, self._callFUT, ICurrent, Current)