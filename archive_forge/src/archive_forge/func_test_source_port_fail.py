from contextlib import closing
import getpass
import os
import socket
import unittest
from tornado.concurrent import Future
from tornado.netutil import bind_sockets, Resolver
from tornado.queues import Queue
from tornado.tcpclient import TCPClient, _Connector
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, gen_test
from tornado.test.util import skipIfNoIPv6, refusing_port, skipIfNonUnix
from tornado.gen import TimeoutError
import typing
@skipIfNonUnix
def test_source_port_fail(self):
    """Fail when trying to use source port 1."""
    if getpass.getuser() == 'root':
        self.skipTest('running as root')
    self.assertRaises(socket.error, self.do_test_connect, socket.AF_INET, '127.0.0.1', source_port=1)