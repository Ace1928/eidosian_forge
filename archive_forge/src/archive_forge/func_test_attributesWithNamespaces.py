from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_attributesWithNamespaces(self):
    """
        Attributes with namespace are parsed without Exception.
        (https://twistedmatrix.com/trac/ticket/9730 regression test)
        """
    xml = b"<root xmlns:test='http://example.org' xml:lang='en'>\n                    <test:test>test</test:test>\n                  </root>"
    self.stream.parse(xml)
    self.assertEqual(self.elements[0].uri, 'http://example.org')