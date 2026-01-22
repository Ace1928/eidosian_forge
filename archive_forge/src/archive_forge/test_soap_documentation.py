from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.trial.unittest import TestCase
from twisted.web import error, server

        Test lookupFunction method on publisher, to see available remote
        methods.
        