import json
import os.path
import shutil
import socket
import ssl
import sys
import tempfile
import warnings
from test import (
import pytest
import trustme
from dummyserver.server import DEFAULT_CA, HAS_IPV6, get_unreachable_address
from dummyserver.testcase import HTTPDummyProxyTestCase, IPv6HTTPDummyProxyTestCase
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import VerifiedHTTPSConnection, connection_from_url
from urllib3.exceptions import (
from urllib3.poolmanager import ProxyManager, proxy_from_url
from urllib3.util import Timeout
from urllib3.util.ssl_ import create_urllib3_context
from .. import TARPIT_HOST, requires_network
@onlyPy3
def test_proxy_verified_warning(self):
    """Skip proxy verification to validate warnings are generated"""
    with warnings.catch_warnings(record=True) as w:
        with proxy_from_url(self.https_proxy_url, cert_reqs='NONE') as https:
            r = https.request('GET', '%s/' % self.https_url)
            assert r.status == 200
    assert len(w) == 2
    assert w[0].category == InsecureRequestWarning
    assert w[1].category == InsecureRequestWarning
    messages = set((str(x.message) for x in w))
    expected = ["Unverified HTTPS request is being made to host 'localhost'", 'Unverified HTTPS connection done to an HTTPS proxy.']
    for warn_message in expected:
        assert [msg for msg in messages if warn_message in expected]