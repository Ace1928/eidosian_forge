import os
import socket
import sys
from unittest import skipIf
from twisted.internet import address, defer, error, interfaces, protocol, reactor, utils
from twisted.python import lockfile
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.test.test_tcp import MyClientFactory, MyServerFactory
from twisted.trial import unittest
def cbConnMade(proto):
    expected = address.UNIXAddress(peername)
    self.assertEqual(serverFactory.peerAddresses, [expected])
    self.assertEqual(proto.transport.getPeer(), expected)