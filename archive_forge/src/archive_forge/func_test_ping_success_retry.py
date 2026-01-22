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
def test_ping_success_retry():
    request = make_request('', headers=_metadata._METADATA_HEADERS, retry=True)
    assert _metadata.ping(request)
    request.assert_called_with(method='GET', url=_metadata._METADATA_IP_ROOT, headers=_metadata._METADATA_HEADERS, timeout=_metadata._METADATA_DEFAULT_TIMEOUT)
    assert request.call_count == 2