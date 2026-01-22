from tornado.concurrent import Future
from tornado import gen
from tornado import netutil
from tornado.ioloop import IOLoop
from tornado.iostream import (
from tornado.httputil import HTTPHeaders
from tornado.locks import Condition, Event
from tornado.log import gen_log
from tornado.netutil import ssl_options_to_context, ssl_wrap_socket
from tornado.platform.asyncio import AddThreadSelectorEventLoop
from tornado.tcpserver import TCPServer
from tornado.testing import (
from tornado.test.util import (
from tornado.web import RequestHandler, Application
import asyncio
import errno
import hashlib
import logging
import os
import platform
import random
import socket
import ssl
import typing
from unittest import mock
import unittest
@gen_test
def test_start_tls_smtp(self):
    yield self.server_send_line(b'220 mail.example.com ready\r\n')
    yield self.client_send_line(b'EHLO mail.example.com\r\n')
    yield self.server_send_line(b'250-mail.example.com welcome\r\n')
    yield self.server_send_line(b'250 STARTTLS\r\n')
    yield self.client_send_line(b'STARTTLS\r\n')
    yield self.server_send_line(b'220 Go ahead\r\n')
    client_future = self.client_start_tls(dict(cert_reqs=ssl.CERT_NONE))
    server_future = self.server_start_tls(_server_ssl_options())
    self.client_stream = (yield client_future)
    self.server_stream = (yield server_future)
    self.assertTrue(isinstance(self.client_stream, SSLIOStream))
    self.assertTrue(isinstance(self.server_stream, SSLIOStream))
    yield self.client_send_line(b'EHLO mail.example.com\r\n')
    yield self.server_send_line(b'250 mail.example.com welcome\r\n')