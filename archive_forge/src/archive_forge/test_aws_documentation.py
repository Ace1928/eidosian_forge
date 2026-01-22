import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
Utility function to generate a mock HTTP request object.
        This will facilitate testing various edge cases by specify how the
        various endpoints will respond while generating a Google Access token
        in an AWS environment.
        