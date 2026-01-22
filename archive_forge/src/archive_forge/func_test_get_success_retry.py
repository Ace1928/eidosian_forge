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
def test_get_success_retry():
    key, value = ('foo', 'bar')
    data = json.dumps({key: value})
    request = make_request(data, headers={'content-type': 'application/json'}, retry=True)
    result = _metadata.get(request, PATH)
    request.assert_called_with(method='GET', url=_metadata._METADATA_ROOT + PATH, headers=_metadata._METADATA_HEADERS)
    assert request.call_count == 2
    assert result[key] == value