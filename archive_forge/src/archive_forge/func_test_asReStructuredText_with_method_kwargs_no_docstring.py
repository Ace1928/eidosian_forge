import sys
import unittest
def test_asReStructuredText_with_method_kwargs_no_docstring(self):
    from zope.interface import Interface
    EXPECTED = '\n\n'.join(['``IHasMethod``', ' This interface has a method.', ' Attributes:', ' Methods:', '  ``aMethod(first, second, **kw)`` -- no documentation', ''])

    class IHasMethod(Interface):
        """ This interface has a method.
            """

        def aMethod(first, second, **kw):
            pass
    self.assertEqual(self._callFUT(IHasMethod), EXPECTED)