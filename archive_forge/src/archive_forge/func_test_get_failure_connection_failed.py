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
def test_get_failure_connection_failed():
    request = make_request('')
    request.side_effect = exceptions.TransportError()
    with pytest.raises(exceptions.TransportError) as excinfo:
        _metadata.get(request, PATH)
    assert excinfo.match('Compute Engine Metadata server unavailable')
    request.assert_called_with(method='GET', url=_metadata._METADATA_ROOT + PATH, headers=_metadata._METADATA_HEADERS)
    assert request.call_count == 5