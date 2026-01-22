from binascii import hexlify
from hashlib import sha1
from sys import intern
from typing import Optional, Tuple
from zope.interface import directlyProvides, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, randbytes
from twisted.words.protocols.jabber import error, ijabber, jid
from twisted.words.xish import domish, xmlstream
from twisted.words.xish.xmlstream import (
def sendHeader(self):
    """
        Send stream header.
        """
    localPrefixes = {}
    for uri, prefix in self.prefixes.items():
        if uri != NS_STREAMS:
            localPrefixes[prefix] = uri
    rootElement = domish.Element((NS_STREAMS, 'stream'), self.namespace, localPrefixes=localPrefixes)
    if self.otherEntity:
        rootElement['to'] = self.otherEntity.userhost()
    if self.thisEntity:
        rootElement['from'] = self.thisEntity.userhost()
    if not self.initiating and self.sid:
        rootElement['id'] = self.sid
    if self.version >= (1, 0):
        rootElement['version'] = '%d.%d' % self.version
    self.send(rootElement.toXml(prefixes=self.prefixes, closeElement=0))
    self._headerSent = True