import pytest
from google.auth import _helpers
from google.auth import exceptions
from google.auth import iam
from google.oauth2 import _service_account_async

    We expect the http request to refresh credentials
    without scopes provided to throw an error.
    