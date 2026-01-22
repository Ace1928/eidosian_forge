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
def get_peername(self):
    """Returns the IP address and port number of the destination machine.

        Note: get_proxy_peername returns the proxy."""
    return self.proxy_peername