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
def test_get_service_account_info():
    key, value = ('foo', 'bar')
    request = make_request(json.dumps({key: value}), headers={'content-type': 'application/json'})
    info = _metadata.get_service_account_info(request)
    request.assert_called_once_with(method='GET', url=_metadata._METADATA_ROOT + PATH + '/?recursive=true', headers=_metadata._METADATA_HEADERS)
    assert info[key] == value