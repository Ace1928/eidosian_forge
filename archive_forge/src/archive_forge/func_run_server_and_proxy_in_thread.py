import collections
import contextlib
import platform
import socket
import ssl
import sys
import threading
import pytest
import trustme
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import HAS_IPV6, run_tornado_app
from dummyserver.testcase import HTTPSDummyServerTestCase
from urllib3.util import ssl_
from .tz_stub import stub_timezone_ctx
@contextlib.contextmanager
def run_server_and_proxy_in_thread(proxy_scheme, proxy_host, tmpdir, ca, proxy_cert, server_cert):
    ca_cert_path = str(tmpdir / 'ca.pem')
    ca.cert_pem.write_to_path(ca_cert_path)
    server_certs = _write_cert_to_dir(server_cert, tmpdir)
    proxy_certs = _write_cert_to_dir(proxy_cert, tmpdir, 'proxy')
    io_loop = ioloop.IOLoop.current()
    server = web.Application([('.*', TestingApp)])
    server, port = run_tornado_app(server, io_loop, server_certs, 'https', 'localhost')
    server_config = ServerConfig('localhost', port, ca_cert_path)
    proxy = web.Application([('.*', ProxyHandler)])
    proxy_app, proxy_port = run_tornado_app(proxy, io_loop, proxy_certs, proxy_scheme, proxy_host)
    proxy_config = ServerConfig(proxy_host, proxy_port, ca_cert_path)
    server_thread = threading.Thread(target=io_loop.start)
    server_thread.start()
    yield (proxy_config, server_config)
    io_loop.add_callback(server.stop)
    io_loop.add_callback(proxy_app.stop)
    io_loop.add_callback(io_loop.stop)
    server_thread.join()