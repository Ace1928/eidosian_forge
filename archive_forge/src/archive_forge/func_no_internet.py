import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
def no_internet():
    """Return true if the NO_INTERNET or the NO_NETWORK environment variable
    is set.
    Can be used to skip relevant tests.
    """
    return 'NO_INTERNET' in os.environ or no_network()