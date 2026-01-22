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
def test_invalid_available_permissions_type(self):
    availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
    with pytest.raises(TypeError) as excinfo:
        make_access_boundary_rule(AVAILABLE_RESOURCE, [0, 1, 2], availability_condition)
    assert excinfo.match('Provided available_permissions are not a list of strings.')