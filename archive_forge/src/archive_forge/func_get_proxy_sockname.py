from base64 import b64encode
import six
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys
def get_proxy_sockname(self):
    """Returns the bound IP address and port number at the proxy."""
    return self.proxy_sockname