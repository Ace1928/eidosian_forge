import base64
import contextlib
import socket
import ssl
import pytest
from ..test_util import TestUtilSSL  # noqa: E402, F401
from ..with_dummyserver.test_https import (  # noqa: E402, F401
from ..with_dummyserver.test_socketlevel import (  # noqa: E402, F401
def test_no_crash_with_invalid_trust_bundle():
    invalid_cert = base64.b64encode(b'invalid-cert')
    cert_bundle = b'-----BEGIN CERTIFICATE-----\n' + invalid_cert + b'\n-----END CERTIFICATE-----'
    with contextlib.closing(socket.socket()) as s:
        ws = WrappedSocket(s)
        with pytest.raises(ssl.SSLError):
            ws._custom_validate(True, cert_bundle)