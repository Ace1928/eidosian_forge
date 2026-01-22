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
def test_manual_ping(self):
    ws = (yield self.ws_connect('/'))
    self.assertRaises(ValueError, ws.ping, 'a' * 126)
    ws.ping('hello')
    resp = (yield ws.read_message())
    self.assertEqual(resp, b'hello')
    ws.ping(b'binary hello')
    resp = (yield ws.read_message())
    self.assertEqual(resp, b'binary hello')