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
def notBrotlipy():
    return pytest.mark.skipif(brotli is not None, reason='only run if brotlipy is absent')