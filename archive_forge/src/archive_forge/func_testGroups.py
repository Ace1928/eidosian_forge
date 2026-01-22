import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
@defer.inlineCallbacks
def testGroups(self):
    mindone = TestMind()
    one = (yield self._loggedInAvatar('one', b'p1', mindone))
    mindtwo = TestMind()
    two = (yield self._loggedInAvatar('two', b'p2', mindtwo))
    mindThree = TestMind()
    three = (yield self._loggedInAvatar(b'three', b'p3', mindThree))
    yield self.realm.createGroup('foobar')
    yield self.realm.createGroup(b'barfoo')
    groupone = (yield one.join('foobar'))
    grouptwo = (yield two.join(b'barfoo'))
    yield two.join('foobar')
    yield two.join(b'barfoo')
    yield three.join('foobar')
    yield groupone.send({b'text': b'hello, monkeys'})
    yield groupone.leave()
    yield grouptwo.leave()