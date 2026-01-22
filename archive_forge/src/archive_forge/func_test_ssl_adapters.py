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
def test_ssl_adapters(http_request_timeout, tls_http_server, adapter_type, tls_certificate, tls_certificate_chain_pem_path, tls_certificate_private_key_pem_path, tls_ca_certificate_pem_path):
    """Test ability to connect to server via HTTPS using adapters."""
    interface, _host, port = _get_conn_data(ANY_INTERFACE_IPV4)
    tls_adapter_cls = get_ssl_adapter_class(name=adapter_type)
    tls_adapter = tls_adapter_cls(tls_certificate_chain_pem_path, tls_certificate_private_key_pem_path)
    if adapter_type == 'pyopenssl':
        tls_adapter.context = tls_adapter.get_context()
    tls_certificate.configure_cert(tls_adapter.context)
    tlshttpserver = tls_http_server((interface, port), tls_adapter)
    interface, _host, port = _get_conn_data(tlshttpserver.bind_addr)
    resp = requests.get('https://{host!s}:{port!s}/'.format(host=interface, port=port), timeout=http_request_timeout, verify=tls_ca_certificate_pem_path)
    assert resp.status_code == 200
    assert resp.text == 'Hello world!'