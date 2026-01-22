import base64
import collections
import json
import os
import os.path
import shlex
import string
from datetime import datetime
from packaging.version import Version
from .. import errors
from ..constants import DEFAULT_HTTP_HOST
from ..constants import DEFAULT_UNIX_SOCKET
from ..constants import DEFAULT_NPIPE
from ..constants import BYTE_UNITS
from ..tls import TLSConfig
from urllib.parse import urlparse, urlunparse
def convert_tmpfs_mounts(tmpfs):
    if isinstance(tmpfs, dict):
        return tmpfs
    if not isinstance(tmpfs, list):
        raise ValueError('Expected tmpfs value to be either a list or a dict, found: {}'.format(type(tmpfs).__name__))
    result = {}
    for mount in tmpfs:
        if isinstance(mount, str):
            if ':' in mount:
                name, options = mount.split(':', 1)
            else:
                name = mount
                options = ''
        else:
            raise ValueError('Expected item in tmpfs list to be a string, found: {}'.format(type(mount).__name__))
        result[name] = options
    return result