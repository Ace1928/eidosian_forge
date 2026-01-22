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
def test_get_success_custom_root_old_variable():
    request = make_request('{}', headers={'content-type': 'application/json'})
    fake_root = 'another.metadata.service'
    os.environ[environment_vars.GCE_METADATA_ROOT] = fake_root
    reload_module(_metadata)
    try:
        _metadata.get(request, PATH)
    finally:
        del os.environ[environment_vars.GCE_METADATA_ROOT]
        reload_module(_metadata)
    request.assert_called_once_with(method='GET', url='http://{}/computeMetadata/v1/{}'.format(fake_root, PATH), headers=_metadata._METADATA_HEADERS)