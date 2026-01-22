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
def hashPassword(sid, password):
    """
    Create a SHA1-digest string of a session identifier and password.

    @param sid: The stream session identifier.
    @type sid: C{unicode}.
    @param password: The password to be hashed.
    @type password: C{unicode}.
    """
    if not isinstance(sid, str):
        raise TypeError('The session identifier must be a unicode object')
    if not isinstance(password, str):
        raise TypeError('The password must be a unicode object')
    input = f'{sid}{password}'
    return sha1(input.encode('utf-8')).hexdigest()