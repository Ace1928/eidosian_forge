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
def sample_request(binary=True):
    request = b'GET http://www.testing.com/ HTTP/1.1\r\nHost: www.testing.com\r\nUser-Agent: awesome-test\r\n\r\n'
    return request if binary else request.decode('utf-8')