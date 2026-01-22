from twisted.application.internet import StreamServerEndpointService
from twisted.application.service import MultiService
from twisted.conch import telnet
from twisted.cred import error
from twisted.cred.credentials import UsernamePassword
from twisted.python import usage
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase

        The C{--passwd} command-line option will load a passwd-like file.
        