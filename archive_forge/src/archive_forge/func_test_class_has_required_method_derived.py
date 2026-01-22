import unittest
def test_class_has_required_method_derived(self):
    from zope.interface import Interface
    from zope.interface import implementer

    class IBase(Interface):

        def method():
            """docstring"""

    class IDerived(IBase):
        pass

    @implementer(IDerived)
    class Current:

        def method(self):
            raise NotImplementedError()
    self._callFUT(IDerived, Current)