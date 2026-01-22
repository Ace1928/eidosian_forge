from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testQualifiedAttributeGivenListOfPrefixes(self):
    e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
    self.assertEqual(e.toXml({'testns2': 'qux'}), "<foo xmlns:qux='testns2' qux:bar='baz'/>")