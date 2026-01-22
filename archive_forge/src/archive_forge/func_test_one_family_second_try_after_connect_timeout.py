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
def test_one_family_second_try_after_connect_timeout(self):
    conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
    self.assert_pending((AF1, 'a'))
    self.resolve_connect(AF1, 'a', False)
    self.assert_pending((AF1, 'b'))
    conn.on_connect_timeout()
    self.connect_futures.pop((AF1, 'b'))
    self.assertTrue(self.streams.pop('b').closed)
    self.assert_pending()
    self.assertEqual(len(conn.streams), 2)
    self.assert_connector_streams_closed(conn)
    self.assertRaises(TimeoutError, future.result)