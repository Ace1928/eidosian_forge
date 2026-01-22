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
def test_invalid_common_name(self):
    with HTTPSConnectionPool('127.0.0.1', self.port, cert_reqs='CERT_REQUIRED', ca_certs=DEFAULT_CA) as https_pool:
        with pytest.raises(MaxRetryError) as e:
            https_pool.request('GET', '/')
        assert isinstance(e.value.reason, SSLError)
        assert "doesn't match" in str(e.value.reason) or 'certificate verify failed' in str(e.value.reason)