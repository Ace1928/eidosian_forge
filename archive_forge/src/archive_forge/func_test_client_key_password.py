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
@requires_ssl_context_keyfile_password
def test_client_key_password(self):
    with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA, key_file=os.path.join(self.certs_dir, PASSWORD_CLIENT_KEYFILE), cert_file=os.path.join(self.certs_dir, CLIENT_CERT), key_password='letmein') as https_pool:
        r = https_pool.request('GET', '/certificate')
        subject = json.loads(r.data.decode('utf-8'))
        assert subject['organizationalUnitName'].startswith('Testing cert')