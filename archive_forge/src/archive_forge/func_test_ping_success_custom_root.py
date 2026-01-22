import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import reload_module
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.auth.compute_engine import _metadata
def test_ping_success_custom_root():
    request = make_request('', headers=_metadata._METADATA_HEADERS)
    fake_ip = '1.2.3.4'
    os.environ[environment_vars.GCE_METADATA_IP] = fake_ip
    reload_module(_metadata)
    try:
        assert _metadata.ping(request)
    finally:
        del os.environ[environment_vars.GCE_METADATA_IP]
        reload_module(_metadata)
    request.assert_called_once_with(method='GET', url='http://' + fake_ip, headers=_metadata._METADATA_HEADERS, timeout=_metadata._METADATA_DEFAULT_TIMEOUT)