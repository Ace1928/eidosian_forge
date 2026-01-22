from test import notPyPy2
import mock
import pytest
from urllib3.exceptions import SNIMissingWarning
from urllib3.util import ssl_
@pytest.mark.parametrize(['ciphers', 'expected_ciphers'], [(None, ssl_.DEFAULT_CIPHERS), ('ECDH+AESGCM:ECDH+CHACHA20', 'ECDH+AESGCM:ECDH+CHACHA20')])
def test_create_urllib3_context_set_ciphers(monkeypatch, ciphers, expected_ciphers):
    context = mock.create_autospec(ssl_.SSLContext)
    context.set_ciphers = mock.Mock()
    context.options = 0
    monkeypatch.setattr(ssl_, 'SSLContext', lambda *_, **__: context)
    assert ssl_.create_urllib3_context(ciphers=ciphers) is context
    assert context.set_ciphers.call_count == 1
    assert context.set_ciphers.call_args == mock.call(expected_ciphers)