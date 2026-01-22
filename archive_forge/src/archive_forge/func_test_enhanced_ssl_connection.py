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
def test_enhanced_ssl_connection(self):
    fingerprint = '72:8B:55:4C:9A:FC:1E:88:A1:1C:AD:1B:B2:E7:CC:3E:DB:C8:F9:8A'
    with HTTPSConnectionPool(self.host, self.port, cert_reqs='CERT_REQUIRED', ca_certs=DEFAULT_CA, assert_fingerprint=fingerprint) as https_pool:
        r = https_pool.request('GET', '/')
        assert r.status == 200