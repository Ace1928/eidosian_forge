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
def test_assert_invalid_fingerprint(self):

    def _test_request(pool):
        with pytest.raises(MaxRetryError) as cm:
            pool.request('GET', '/', retries=0)
        assert isinstance(cm.value.reason, SSLError)
        return cm.value.reason
    with HTTPSConnectionPool(self.host, self.port, cert_reqs='CERT_REQUIRED', ca_certs=DEFAULT_CA) as https_pool:
        https_pool.assert_fingerprint = 'AA:AA:AA:AA:AA:AAAA:AA:AAAA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA'
        e = _test_request(https_pool)
        assert 'Fingerprints did not match.' in str(e)
        https_pool.assert_fingerprint = 'AA:A'
        e = _test_request(https_pool)
        assert 'Fingerprint of invalid length:' in str(e)
        https_pool.assert_fingerprint = 'AA'
        e = _test_request(https_pool)
        assert 'Fingerprint of invalid length:' in str(e)