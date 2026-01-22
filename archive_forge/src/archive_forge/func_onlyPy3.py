import errno
import logging
import os
import platform
import socket
import ssl
import sys
import warnings
import pytest
from urllib3 import util
from urllib3.exceptions import HTTPWarning
from urllib3.packages import six
from urllib3.util import ssl_
def onlyPy3(test):
    """Skips this test unless you are on Python3.x"""

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        msg = '{name} requires Python3.x to run'.format(name=test.__name__)
        if six.PY2:
            pytest.skip(msg)
        return test(*args, **kwargs)
    return wrapper