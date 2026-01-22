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
def tls_certificate_private_key_pem_path(tls_certificate):
    """Provide a certificate private key PEM file path via fixture."""
    with tls_certificate.private_key_pem.tempfile() as cert_key_pem:
        yield cert_key_pem