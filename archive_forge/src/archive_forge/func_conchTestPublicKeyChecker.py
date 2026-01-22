import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
def conchTestPublicKeyChecker():
    """
        Produces a SSHPublicKeyChecker with an in-memory key mapping with
        a single use: 'testuser'

        @return: L{twisted.conch.checkers.SSHPublicKeyChecker}
        """
    conchTestPublicKeyDB = checkers.InMemorySSHKeyDB({b'testuser': [keys.Key.fromString(publicDSA_openssh)]})
    return checkers.SSHPublicKeyChecker(conchTestPublicKeyDB)