import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase

        Every public class in twisted.mail._pop3client should be available as
        a member of twisted.mail.pop3 with the exception of
        twisted.mail._pop3client.POP3Client which should be available as
        twisted.mail.pop3.AdvancedClient.
        