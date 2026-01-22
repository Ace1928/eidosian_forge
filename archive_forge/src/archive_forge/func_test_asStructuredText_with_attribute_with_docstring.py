import sys
import unittest
def test_asStructuredText_with_attribute_with_docstring(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    EXPECTED = '\n\n'.join(['IHasAttribute', ' This interface has an attribute.', ' Attributes:', '  an_attribute -- This attribute is documented.', ' Methods:', ''])

    class IHasAttribute(Interface):
        """ This interface has an attribute.
            """
        an_attribute = Attribute('an_attribute', 'This attribute is documented.')
    self.assertEqual(self._callFUT(IHasAttribute), EXPECTED)