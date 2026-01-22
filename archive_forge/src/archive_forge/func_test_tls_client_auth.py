import functools
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import traceback
import http.client
import OpenSSL.SSL
import pytest
import requests
import trustme
from .._compat import bton, ntob, ntou
from .._compat import IS_ABOVE_OPENSSL10, IS_CI, IS_PYPY
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS
from ..server import HTTPServer, get_ssl_adapter_class
from ..testing import (
from ..wsgi import Gateway_10
@pytest.mark.parametrize('adapter_type', ('builtin', 'pyopenssl'))
@pytest.mark.parametrize(('is_trusted_cert', 'tls_client_identity'), ((True, 'localhost'), (True, '127.0.0.1'), (True, '*.localhost'), (True, 'not_localhost'), (False, 'localhost')))
@pytest.mark.parametrize('tls_verify_mode', (ssl.CERT_NONE, ssl.CERT_OPTIONAL, ssl.CERT_REQUIRED))
@pytest.mark.xfail(IS_PYPY and IS_CI, reason='Fails under PyPy in CI for unknown reason', strict=False)
def test_tls_client_auth(http_request_timeout, mocker, tls_http_server, adapter_type, ca, tls_certificate, tls_certificate_chain_pem_path, tls_certificate_private_key_pem_path, tls_ca_certificate_pem_path, is_trusted_cert, tls_client_identity, tls_verify_mode):
    """Verify that client TLS certificate auth works correctly."""
    test_cert_rejection = tls_verify_mode != ssl.CERT_NONE and (not is_trusted_cert)
    interface, _host, port = _get_conn_data(ANY_INTERFACE_IPV4)
    client_cert_root_ca = ca if is_trusted_cert else trustme.CA()
    with mocker.mock_module.patch('idna.core.ulabel', return_value=ntob(tls_client_identity)):
        client_cert = client_cert_root_ca.issue_cert(ntou(tls_client_identity))
        del client_cert_root_ca
    with client_cert.private_key_and_cert_chain_pem.tempfile() as cl_pem:
        tls_adapter_cls = get_ssl_adapter_class(name=adapter_type)
        tls_adapter = tls_adapter_cls(tls_certificate_chain_pem_path, tls_certificate_private_key_pem_path)
        if adapter_type == 'pyopenssl':
            tls_adapter.context = tls_adapter.get_context()
            tls_adapter.context.set_verify(_stdlib_to_openssl_verify[tls_verify_mode], lambda conn, cert, errno, depth, preverify_ok: preverify_ok)
        else:
            tls_adapter.context.verify_mode = tls_verify_mode
        ca.configure_trust(tls_adapter.context)
        tls_certificate.configure_cert(tls_adapter.context)
        tlshttpserver = tls_http_server((interface, port), tls_adapter)
        interface, _host, port = _get_conn_data(tlshttpserver.bind_addr)
        make_https_request = functools.partial(requests.get, 'https://{host!s}:{port!s}/'.format(host=interface, port=port), timeout=http_request_timeout, verify=tls_ca_certificate_pem_path, cert=cl_pem)
        if not test_cert_rejection:
            resp = make_https_request()
            is_req_successful = resp.status_code == 200
            if not is_req_successful and IS_PYOPENSSL_SSL_VERSION_1_0 and (adapter_type == 'builtin') and (tls_verify_mode == ssl.CERT_REQUIRED) and (tls_client_identity == 'localhost') and is_trusted_cert:
                pytest.xfail('OpenSSL 1.0 has problems with verifying client certs')
            assert is_req_successful
            assert resp.text == 'Hello world!'
            resp.close()
            return
        issue_237 = IS_MACOS and adapter_type == 'builtin' and (tls_verify_mode != ssl.CERT_NONE)
        if issue_237:
            pytest.xfail('Test sometimes fails')
        expected_ssl_errors = (requests.exceptions.SSLError,)
        if IS_WINDOWS or IS_GITHUB_ACTIONS_WORKFLOW:
            expected_ssl_errors += (requests.exceptions.ConnectionError,)
        with pytest.raises(expected_ssl_errors) as ssl_err:
            make_https_request().close()
        try:
            err_text = ssl_err.value.args[0].reason.args[0].args[0]
        except AttributeError:
            if IS_WINDOWS or IS_GITHUB_ACTIONS_WORKFLOW:
                err_text = str(ssl_err.value)
            else:
                raise
        if isinstance(err_text, int):
            err_text = str(ssl_err.value)
        expected_substrings = ('sslv3 alert bad certificate' if IS_LIBRESSL_BACKEND else 'tlsv1 alert unknown ca',)
        if IS_MACOS and IS_PYPY and (adapter_type == 'pyopenssl'):
            expected_substrings = ('tlsv1 alert unknown ca',)
        if tls_verify_mode in (ssl.CERT_REQUIRED, ssl.CERT_OPTIONAL) and (not is_trusted_cert) and (tls_client_identity == 'localhost'):
            expected_substrings += ("bad handshake: SysCallError(10054, 'WSAECONNRESET')", '(\'Connection aborted.\', OSError("(10054, \'WSAECONNRESET\')"))', '(\'Connection aborted.\', OSError("(10054, \'WSAECONNRESET\')",))', '(\'Connection aborted.\', error("(10054, \'WSAECONNRESET\')",))', "('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))", "('Connection aborted.', error(10054, 'An existing connection was forcibly closed by the remote host'))") if IS_WINDOWS else ('(\'Connection aborted.\', OSError("(104, \'ECONNRESET\')"))', '(\'Connection aborted.\', OSError("(104, \'ECONNRESET\')",))', '(\'Connection aborted.\', error("(104, \'ECONNRESET\')",))', "('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))", "('Connection aborted.', error(104, 'Connection reset by peer'))") if IS_GITHUB_ACTIONS_WORKFLOW and IS_LINUX else ("('Connection aborted.', BrokenPipeError(32, 'Broken pipe'))",)
        if PY310_PLUS:
            expected_substrings += ("SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:",)
        if IS_GITHUB_ACTIONS_WORKFLOW and IS_WINDOWS and PY310_PLUS:
            expected_substrings += ("('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))",)
        assert any((e in err_text for e in expected_substrings))