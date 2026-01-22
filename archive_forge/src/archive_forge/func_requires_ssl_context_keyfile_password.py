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
def requires_ssl_context_keyfile_password(test):

    @six.wraps(test)
    def wrapper(*args, **kwargs):
        if not ssl_.IS_PYOPENSSL and sys.version_info < (2, 7, 9) or ssl_.IS_SECURETRANSPORT:
            pytest.skip('%s requires password parameter for SSLContext.load_cert_chain()' % test.__name__)
        return test(*args, **kwargs)
    return wrapper