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
def make_availability_condition(expression, title=None, description=None):
    return downscoped.AvailabilityCondition(expression, title, description)