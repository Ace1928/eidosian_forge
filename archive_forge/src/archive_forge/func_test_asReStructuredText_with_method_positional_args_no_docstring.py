import sys
import unittest
def test_asReStructuredText_with_method_positional_args_no_docstring(self):
    from zope.interface import Interface
    EXPECTED = '\n\n'.join(['``IHasMethod``', ' This interface has a method.', ' Attributes:', ' Methods:', '  ``aMethod(first, second)`` -- no documentation', ''])

    class IHasMethod(Interface):
        """ This interface has a method.
            """

        def aMethod(first, second):
            pass
    self.assertEqual(self._callFUT(IHasMethod), EXPECTED)