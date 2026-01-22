import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testGroupMessage(self):
    user = self._loggedInUser('useruser')
    self.successResultOf(self.realm.createGroup('somechannel'))
    user.write('JOIN #somechannel\r\n')
    other = self._loggedInUser('otheruser')
    other.write('JOIN #somechannel\r\n')
    user.transport.clear()
    other.transport.clear()
    user.write('PRIVMSG #somechannel :Hello, world.\r\n')
    response = self._response(user)
    event = self._response(other)
    self.assertFalse(response)
    self.assertEqual(len(event), 1)
    self.assertEqual(event[0][0], 'useruser!useruser@realmname')
    self.assertEqual(event[0][1], 'PRIVMSG', -1)
    self.assertEqual(event[0][2], ['#somechannel', 'Hello, world.'])