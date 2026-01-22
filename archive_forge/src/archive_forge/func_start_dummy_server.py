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
def start_dummy_server(self, handler=None):

    def socket_handler(listener):
        sock = listener.accept()[0]
        with self.server_context.wrap_socket(sock, server_side=True) as ssock:
            request = consume_socket(ssock)
            validate_request(request)
            ssock.send(sample_response())
    chosen_handler = handler if handler else socket_handler
    self._start_server(chosen_handler)