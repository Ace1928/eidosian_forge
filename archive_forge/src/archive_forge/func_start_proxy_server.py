import platform
import select
import socket
import ssl
import sys
import mock
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3.util import ssl_
from urllib3.util.ssltransport import SSLTransport
@classmethod
def start_proxy_server(cls):
    cls.proxy_server = SocketProxyDummyServer(cls.host, cls.port)
    cls.proxy_server.start_proxy_handler()