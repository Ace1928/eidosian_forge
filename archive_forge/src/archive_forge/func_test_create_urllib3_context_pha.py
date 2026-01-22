from test import notPyPy2
import mock
import pytest
from urllib3.exceptions import SNIMissingWarning
from urllib3.util import ssl_
@pytest.mark.parametrize(['pha', 'expected_pha'], [(None, None), (False, True), (True, True)])
def test_create_urllib3_context_pha(monkeypatch, pha, expected_pha):
    context = mock.create_autospec(ssl_.SSLContext)
    context.set_ciphers = mock.Mock()
    context.options = 0
    context.post_handshake_auth = pha
    monkeypatch.setattr(ssl_, 'SSLContext', lambda *_, **__: context)
    assert ssl_.create_urllib3_context() is context
    assert context.post_handshake_auth == expected_pha