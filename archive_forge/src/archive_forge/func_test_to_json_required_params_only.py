import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials
from google.auth import downscoped
from google.auth import exceptions
from google.auth import transport
def test_to_json_required_params_only(self):
    access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS)
    assert access_boundary_rule.to_json() == {'availablePermissions': AVAILABLE_PERMISSIONS, 'availableResource': AVAILABLE_RESOURCE}