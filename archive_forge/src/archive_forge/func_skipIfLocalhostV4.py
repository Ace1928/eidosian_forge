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
def skipIfLocalhostV4(self):
    addrinfo = self.io_loop.run_sync(lambda: Resolver().resolve('localhost', 80))
    families = set((addr[0] for addr in addrinfo))
    if socket.AF_INET6 not in families:
        self.skipTest('localhost does not resolve to ipv6')