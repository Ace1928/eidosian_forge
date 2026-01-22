import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
def header_has_value(header_name, header_value, headers):
    """Check that a header with a given value is present."""
    return header_name.lower() in (k.lower() for k, v in headers if v == header_value)