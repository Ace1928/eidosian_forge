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
def test_all_fail(self):
    conn, future = self.start_connect(self.addrinfo)
    self.assert_pending((AF1, 'a'))
    conn.on_timeout()
    self.assert_pending((AF1, 'a'), (AF2, 'c'))
    self.resolve_connect(AF2, 'c', False)
    self.assert_pending((AF1, 'a'), (AF2, 'd'))
    self.resolve_connect(AF2, 'd', False)
    self.assert_pending((AF1, 'a'))
    self.resolve_connect(AF1, 'a', False)
    self.assert_pending((AF1, 'b'))
    self.assertFalse(future.done())
    self.resolve_connect(AF1, 'b', False)
    self.assertRaises(IOError, future.result)