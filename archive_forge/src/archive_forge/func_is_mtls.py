from __future__ import absolute_import
import functools
import logging
import numbers
import os
import time
import requests.adapters  # pylint: disable=ungrouped-imports
import requests.exceptions  # pylint: disable=ungrouped-imports
from requests.packages.urllib3.util.ssl_ import (  # type: ignore
import six  # pylint: disable=ungrouped-imports
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
import google.auth.transport._mtls_helper
from google.oauth2 import service_account
@property
def is_mtls(self):
    """Indicates if the created SSL channel is mutual TLS."""
    return self._is_mtls