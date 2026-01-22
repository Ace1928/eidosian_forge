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
Utility to assert that a credentials are initialized with the expected
        attributes by calling refresh functionality and confirming response matches
        expected one and that the underlying requests were populated with the
        expected parameters.
        