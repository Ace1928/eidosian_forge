import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testEnumeration(self):
    realm = service.InMemoryWordsRealm('realmname')
    self.successResultOf(realm.createGroup('groupone'))
    self.successResultOf(realm.createGroup('grouptwo'))
    groups = self.successResultOf(realm.itergroups())
    n = [g.name for g in groups]
    n.sort()
    self.assertEqual(n, ['groupone', 'grouptwo'])