from concurrent import futures
import logging
import re
import socket
import typing
import unittest
from tornado.concurrent import (
from tornado.escape import utf8, to_unicode
from tornado import gen
from tornado.iostream import IOStream
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
def test_generator_error(self: typing.Any):

    @gen.coroutine
    def f():
        with self.assertRaisesRegex(CapError, 'already capitalized'):
            yield self.client.capitalize('HELLO')
    self.io_loop.run_sync(f)