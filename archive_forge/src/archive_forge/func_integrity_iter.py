import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def integrity_iter(iter, checksum):
    """Check image data integrity.

    :raises: IOError
    """
    try:
        md5sum = hashlib.new('md5')
    except ValueError:
        raise IOError(errno.EPIPE, 'Corrupt image download. Expected checksum is %s but md5 algorithm is not available on the client' % checksum)
    for chunk in iter:
        yield chunk
        if isinstance(chunk, str):
            chunk = bytes(chunk, 'latin-1')
        md5sum.update(chunk)
    md5sum = md5sum.hexdigest()
    if md5sum != checksum:
        raise IOError(errno.EPIPE, 'Corrupt image download. Checksum was %s expected %s' % (md5sum, checksum))