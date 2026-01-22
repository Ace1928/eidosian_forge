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
def test_unverified_ssl(self):
    """Test that bare HTTPSConnection can connect, make requests"""
    with HTTPSConnectionPool(self.host, self.port, cert_reqs=ssl.CERT_NONE) as pool:
        with mock.patch('warnings.warn') as warn:
            r = pool.request('GET', '/')
            assert r.status == 200
            assert warn.called
            calls = warn.call_args_list
            assert InsecureRequestWarning in [x[0][1] for x in calls]