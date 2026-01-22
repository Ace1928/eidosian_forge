import sys
import unittest
def test_asStructuredText_with_method_no_args_no_docstring(self):
    from zope.interface import Interface
    EXPECTED = '\n\n'.join(['IHasMethod', ' This interface has a method.', ' Attributes:', ' Methods:', '  aMethod() -- no documentation', ''])

    class IHasMethod(Interface):
        """ This interface has a method.
            """

        def aMethod():
            pass
    self.assertEqual(self._callFUT(IHasMethod), EXPECTED)