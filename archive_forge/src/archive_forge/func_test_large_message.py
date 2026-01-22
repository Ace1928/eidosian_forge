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
def test_large_message(self):
    ws = (yield self.ws_connect('/'))
    msg = 'a' * 1024
    ws.write_message(msg)
    resp = (yield ws.read_message())
    self.assertEqual(resp, msg)
    ws.write_message(msg + 'b')
    resp = (yield ws.read_message())
    self.assertIs(resp, None)
    self.assertEqual(ws.close_code, 1009)
    self.assertEqual(ws.close_reason, 'message too big')