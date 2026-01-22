import collections
import contextlib
import platform
import socket
import ssl
import sys
import threading
import pytest
import trustme
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import HAS_IPV6, run_tornado_app
from dummyserver.testcase import HTTPSDummyServerTestCase
from urllib3.util import ssl_
from .tz_stub import stub_timezone_ctx
@pytest.fixture(scope='function')
def requires_tlsv1_3(supported_tls_versions):
    """Test requires TLSv1.3 available"""
    if not getattr(ssl, 'HAS_TLSv1_3', False) or 'TLSv1.3' not in supported_tls_versions:
        pytest.skip('Test requires TLSv1.3')