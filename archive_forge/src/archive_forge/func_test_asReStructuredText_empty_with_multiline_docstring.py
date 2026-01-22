import sys
import unittest
def test_asReStructuredText_empty_with_multiline_docstring(self):
    from zope.interface import Interface
    indent = ' ' * 12 if sys.version_info < (3, 13) else ''
    EXPECTED = '\n'.join(['``IEmpty``', '', ' This is an empty interface.', ' ', f'{indent} It can be used to annotate any class or object, because it promises', f'{indent} nothing.', '', ' Attributes:', '', ' Methods:', '', ''])

    class IEmpty(Interface):
        """ This is an empty interface.

            It can be used to annotate any class or object, because it promises
            nothing.
            """
    self.assertEqual(self._callFUT(IEmpty), EXPECTED)