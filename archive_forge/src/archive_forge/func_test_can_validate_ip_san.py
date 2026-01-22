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
def test_can_validate_ip_san(self, ipv4_san_server):
    """Ensure that urllib3 can validate SANs with IP addresses in them."""
    with HTTPSConnectionPool(ipv4_san_server.host, ipv4_san_server.port, cert_reqs='CERT_REQUIRED', ca_certs=ipv4_san_server.ca_certs) as https_pool:
        r = https_pool.request('GET', '/')
        assert r.status == 200