from test import notPyPy2
import mock
import pytest
from urllib3.exceptions import SNIMissingWarning
from urllib3.util import ssl_
def test_wrap_socket_given_context_no_load_default_certs():
    context = mock.create_autospec(ssl_.SSLContext)
    context.load_default_certs = mock.Mock()
    sock = mock.Mock()
    ssl_.ssl_wrap_socket(sock, ssl_context=context)
    context.load_default_certs.assert_not_called()