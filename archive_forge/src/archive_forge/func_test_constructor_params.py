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
def test_constructor_params(self):
    server_hostname = 'example-domain.com'
    sock = mock.Mock()
    context = mock.create_autospec(ssl_.SSLContext)
    ssl_transport = SSLTransport(sock, context, server_hostname=server_hostname, suppress_ragged_eofs=False)
    context.wrap_bio.assert_called_with(mock.ANY, mock.ANY, server_hostname=server_hostname)
    assert not ssl_transport.suppress_ragged_eofs