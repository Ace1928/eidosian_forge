from test import notPyPy2
import mock
import pytest
from urllib3.exceptions import SNIMissingWarning
from urllib3.util import ssl_
def test_wrap_socket_default_loads_default_certs(monkeypatch):
    context = mock.create_autospec(ssl_.SSLContext)
    context.load_default_certs = mock.Mock()
    context.options = 0
    monkeypatch.setattr(ssl_, 'SSLContext', lambda *_, **__: context)
    sock = mock.Mock()
    ssl_.ssl_wrap_socket(sock)
    context.load_default_certs.assert_called_with()