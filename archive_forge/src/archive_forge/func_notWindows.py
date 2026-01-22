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
def notWindows(test):
    """Skips this test on Windows"""

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        msg = '{name} does not run on Windows'.format(name=test.__name__)
        if platform.system() == 'Windows':
            pytest.skip(msg)
        return test(*args, **kwargs)
    return wrapper