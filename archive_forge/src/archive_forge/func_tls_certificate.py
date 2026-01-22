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
@pytest.fixture
def tls_certificate(ca):
    """Provide a leaf certificate via fixture."""
    interface, _host, _port = _get_conn_data(ANY_INTERFACE_IPV4)
    return ca.issue_cert(ntou(interface))