import asyncio
import contextlib
import functools
import socket
import traceback
import typing
import unittest
from tornado.concurrent import Future
from tornado import gen
from tornado.httpclient import HTTPError, HTTPRequest
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, gen_test, bind_unused_port, ExpectLog
from tornado.web import Application, RequestHandler
from tornado.websocket import (
@gen_test
def test_size_limit(self: typing.Any):
    ws = (yield self.ws_connect('/limited', compression_options=self.get_client_compression_options()))
    ws.write_message('a' * 128)
    response = (yield ws.read_message())
    self.assertEqual(response, '128')
    ws.write_message('a' * 2048)
    response = (yield ws.read_message())
    self.assertIsNone(response)