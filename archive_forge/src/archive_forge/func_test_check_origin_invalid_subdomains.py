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
def test_check_origin_invalid_subdomains(self):
    port = self.get_http_port()
    addrinfo = (yield Resolver().resolve('localhost', port))
    families = set((addr[0] for addr in addrinfo))
    if socket.AF_INET not in families:
        self.skipTest('localhost does not resolve to ipv4')
        return
    url = 'ws://localhost:%d/echo' % port
    headers = {'Origin': 'http://subtenant.localhost'}
    with self.assertRaises(HTTPError) as cm:
        yield websocket_connect(HTTPRequest(url, headers=headers))
    self.assertEqual(cm.exception.code, 403)