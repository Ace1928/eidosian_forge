from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testUnicodeSerialization(self):
    e = domish.Element((None, 'foo'))
    e['test'] = 'my valueȡe'
    e.addContent('A degree symbol...°')
    self.assertEqual(e.toXml(), "<foo test='my valueȡe'>A degree symbol...°</foo>")