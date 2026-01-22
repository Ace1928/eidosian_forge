import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testSetTopic(self):
    user = self._loggedInUser('useruser')
    somechannel = self.successResultOf(self.realm.createGroup('somechannel'))
    user.write('JOIN #somechannel\r\n')
    other = self._loggedInUser('otheruser')
    other.write('JOIN #somechannel\r\n')
    user.transport.clear()
    other.transport.clear()
    other.write('TOPIC #somechannel :This is the new topic.\r\n')
    response = self._response(other)
    event = self._response(user)
    self.assertEqual(response, event)
    self.assertEqual(response[0][0], 'otheruser!otheruser@realmname')
    self.assertEqual(response[0][1], 'TOPIC')
    self.assertEqual(response[0][2], ['#somechannel', 'This is the new topic.'])
    other.transport.clear()
    somechannel.meta['topic_date'] = 12345
    other.write('TOPIC #somechannel\r\n')
    response = self._response(other)
    self.assertEqual(response[0][1], '332')
    self.assertEqual(response[0][2], ['otheruser', '#somechannel', 'This is the new topic.'])
    self.assertEqual(response[1][1], '333')
    self.assertEqual(response[1][2], ['otheruser', '#somechannel', 'otheruser', '12345'])
    other.transport.clear()
    other.write('TOPIC #asdlkjasd\r\n')
    response = self._response(other)
    self.assertEqual(response[0][1], '403')