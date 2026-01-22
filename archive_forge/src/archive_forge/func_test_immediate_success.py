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
def test_immediate_success(self):
    conn, future = self.start_connect(self.addrinfo)
    self.assertEqual(list(self.connect_futures.keys()), [(AF1, 'a')])
    self.resolve_connect(AF1, 'a', True)
    self.assertEqual(future.result(), (AF1, 'a', self.streams['a']))