from test import notPyPy2
import mock
import pytest
from urllib3.exceptions import SNIMissingWarning
from urllib3.util import ssl_
@notPyPy2
def test_wrap_socket_given_ca_certs_no_load_default_certs(monkeypatch):
    context = mock.create_autospec(ssl_.SSLContext)
    context.load_default_certs = mock.Mock()
    context.options = 0
    monkeypatch.setattr(ssl_, 'SSLContext', lambda *_, **__: context)
    sock = mock.Mock()
    ssl_.ssl_wrap_socket(sock, ca_certs='/tmp/fake-file')
    context.load_default_certs.assert_not_called()
    context.load_verify_locations.assert_called_with('/tmp/fake-file', None, None)