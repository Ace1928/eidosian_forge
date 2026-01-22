from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer

        rport is allowed without a value, apparently because server
        implementors might be too stupid to check the received port
        against 5060 and see if they're equal, and because client
        implementors might be too stupid to bind to port 5060, or set a
        value on the rport parameter they send if they bind to another
        port.
        