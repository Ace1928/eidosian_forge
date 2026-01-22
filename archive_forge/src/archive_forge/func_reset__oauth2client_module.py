import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
@pytest.fixture
def reset__oauth2client_module():
    """Reloads the _oauth2client module after a test."""
    reload_module(_oauth2client)