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
@pytest.mark.timeout(PER_TEST_TIMEOUT)
def test_close_after_handshake(self):
    """Socket errors should be bubbled up"""
    self.start_dummy_server()
    sock = socket.create_connection((self.host, self.port))
    with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
        ssock.close()
        with pytest.raises(OSError):
            ssock.send(b'blaaargh')