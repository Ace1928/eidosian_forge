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
def get_default_proxy():
    """Returns the default proxy, set by set_default_proxy."""
    return socksocket.default_proxy