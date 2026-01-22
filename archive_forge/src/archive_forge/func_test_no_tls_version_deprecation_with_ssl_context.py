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
def test_no_tls_version_deprecation_with_ssl_context(self):
    if self.tls_protocol_name is None:
        pytest.skip('Skipping base test class')
    with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA, ssl_context=util.ssl_.create_urllib3_context()) as https_pool:
        conn = https_pool._get_conn()
        try:
            with warnings.catch_warnings(record=True) as w:
                conn.connect()
        finally:
            conn.close()
    assert w == []