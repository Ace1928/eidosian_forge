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
@pytest.mark.parametrize('ip_addr', (ANY_INTERFACE_IPV4, ANY_INTERFACE_IPV6))
def test_https_over_http_error(http_server, ip_addr):
    """Ensure that connecting over HTTPS to HTTP port is handled."""
    httpserver = http_server.send((ip_addr, EPHEMERAL_PORT))
    interface, _host, port = _get_conn_data(httpserver.bind_addr)
    with pytest.raises(ssl.SSLError) as ssl_err:
        http.client.HTTPSConnection('{interface}:{port}'.format(interface=interface, port=port)).request('GET', '/')
    expected_substring = 'wrong version number' if IS_ABOVE_OPENSSL10 else 'unknown protocol'
    assert expected_substring in ssl_err.value.args[-1]