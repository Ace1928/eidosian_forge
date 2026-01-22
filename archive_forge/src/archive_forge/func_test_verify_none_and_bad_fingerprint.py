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
def test_verify_none_and_bad_fingerprint(self):
    with HTTPSConnectionPool('127.0.0.1', self.port, cert_reqs='CERT_NONE', ca_certs=self.bad_ca_path) as https_pool:
        https_pool.assert_fingerprint = 'AA:AA:AA:AA:AA:AAAA:AA:AAAA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA'
        with pytest.raises(MaxRetryError) as cm:
            https_pool.request('GET', '/', retries=0)
        assert isinstance(cm.value.reason, SSLError)