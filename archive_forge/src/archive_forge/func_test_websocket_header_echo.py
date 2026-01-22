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
def test_websocket_header_echo(self):
    with contextlib.closing((yield websocket_connect(HTTPRequest('ws://127.0.0.1:%d/header_echo' % self.get_http_port(), headers={'X-Test-Hello': 'hello'})))) as ws:
        self.assertEqual(ws.headers.get('X-Test-Hello'), 'hello')
        self.assertEqual(ws.headers.get('X-Extra-Response-Header'), 'Extra-Response-Value')