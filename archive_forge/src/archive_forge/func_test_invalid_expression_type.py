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
def test_invalid_expression_type(self):
    with pytest.raises(TypeError) as excinfo:
        make_availability_condition([EXPRESSION], TITLE, DESCRIPTION)
    assert excinfo.match('The provided expression is not a string.')