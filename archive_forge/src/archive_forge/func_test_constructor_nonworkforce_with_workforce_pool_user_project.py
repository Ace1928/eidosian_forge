import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import identity_pool
from google.auth import transport
def test_constructor_nonworkforce_with_workforce_pool_user_project(self):
    with pytest.raises(ValueError) as excinfo:
        self.make_credentials(audience=AUDIENCE, workforce_pool_user_project=WORKFORCE_POOL_USER_PROJECT)
    assert excinfo.match('workforce_pool_user_project should not be set for non-workforce pool credentials')