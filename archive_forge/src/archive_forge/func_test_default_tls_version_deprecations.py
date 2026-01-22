import datetime
import json
import logging
import os.path
import shutil
import ssl
import sys
import tempfile
import warnings
from test import (
import mock
import pytest
import trustme
import urllib3.util as util
from dummyserver.server import (
from dummyserver.testcase import HTTPSDummyServerTestCase
from urllib3 import HTTPSConnectionPool
from urllib3.connection import RECENT_DATE, VerifiedHTTPSConnection
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.util.timeout import Timeout
from .. import has_alpn
def test_default_tls_version_deprecations(self):
    if self.tls_protocol_name is None:
        pytest.skip('Skipping base test class')
    with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as https_pool:
        conn = https_pool._get_conn()
        try:
            with warnings.catch_warnings(record=True) as w:
                conn.connect()
                if not hasattr(conn.sock, 'version'):
                    pytest.skip('SSLSocket.version() not available')
        finally:
            conn.close()
    if self.tls_protocol_deprecated():
        assert len(w) == 1
        assert str(w[0].message) == "Negotiating TLSv1/TLSv1.1 by default is deprecated and will be disabled in urllib3 v2.0.0. Connecting to '%s' with '%s' can be enabled by explicitly opting-in with 'ssl_version'" % (self.host, self.tls_protocol_name)
    else:
        assert w == []