import sys
import unittest
def test_asStructuredText_no_docstring(self):
    from zope.interface import Interface
    EXPECTED = '\n\n'.join(['INoDocstring', ' Attributes:', ' Methods:', ''])

    class INoDocstring(Interface):
        pass
    self.assertEqual(self._callFUT(INoDocstring), EXPECTED)