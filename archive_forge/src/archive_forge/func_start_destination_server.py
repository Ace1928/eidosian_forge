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
def start_destination_server(cls):
    """
        Socket handler for the destination_server. Terminates the second TLS
        layer and send a basic HTTP response.
        """

    def socket_handler(listener):
        sock = listener.accept()[0]
        with cls.server_context.wrap_socket(sock, server_side=True) as ssock:
            request = consume_socket(ssock)
            validate_request(request)
            ssock.send(sample_response())
        sock.close()
    cls._start_server(socket_handler)