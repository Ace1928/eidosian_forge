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
def test_ping_failure_bad_flavor():
    request = make_request('', headers={_metadata._METADATA_FLAVOR_HEADER: 'meep'})
    assert not _metadata.ping(request)