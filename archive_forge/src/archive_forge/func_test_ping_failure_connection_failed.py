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
def test_ping_failure_connection_failed():
    request = make_request('')
    request.side_effect = exceptions.TransportError()
    assert not _metadata.ping(request)