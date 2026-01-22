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
def withPyOpenSSL(test):

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        if not pyopenssl:
            pytest.skip('pyopenssl not available, skipping test.')
            return test(*args, **kwargs)
        pyopenssl.inject_into_urllib3()
        result = test(*args, **kwargs)
        pyopenssl.extract_from_urllib3()
        return result
    return wrapper