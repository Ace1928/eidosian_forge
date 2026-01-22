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
def resolve_connect(self, af, addr, success):
    future = self.connect_futures.pop((af, addr))
    if success:
        future.set_result(self.streams[addr])
    else:
        self.streams.pop(addr)
        future.set_exception(IOError())
    self.io_loop.add_callback(self.stop)
    self.wait()