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
def test_get_project_id():
    project = 'example-project'
    request = make_request(project, headers={'content-type': 'text/plain'})
    project_id = _metadata.get_project_id(request)
    request.assert_called_once_with(method='GET', url=_metadata._METADATA_ROOT + 'project/project-id', headers=_metadata._METADATA_HEADERS)
    assert project_id == project