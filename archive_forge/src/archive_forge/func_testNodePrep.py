from twisted.trial import unittest
from twisted.words.protocols.jabber.xmpp_stringprep import (
def testNodePrep(self) -> None:
    self.assertEqual(nodeprep.prepare('user'), 'user')
    self.assertEqual(nodeprep.prepare('User'), 'user')
    self.assertRaises(UnicodeError, nodeprep.prepare, 'us&er')