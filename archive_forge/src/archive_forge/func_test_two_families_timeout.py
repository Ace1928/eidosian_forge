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
def test_two_families_timeout(self):
    conn, future = self.start_connect(self.addrinfo)
    self.assert_pending((AF1, 'a'))
    conn.on_timeout()
    self.assert_pending((AF1, 'a'), (AF2, 'c'))
    self.resolve_connect(AF2, 'c', True)
    self.assertEqual(future.result(), (AF2, 'c', self.streams['c']))
    self.resolve_connect(AF1, 'a', False)
    self.assert_pending()