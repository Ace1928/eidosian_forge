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
def serious_integrity_iter(iter, hasher, hash_value):
    """Check image data integrity using the Glance "multihash".

    :param iter: iterable containing the image data
    :param hasher: a hashlib object
    :param hash_value: hexdigest of the image data
    :raises: IOError if the hashdigest of the data is not hash_value
    """
    for chunk in iter:
        yield chunk
        if isinstance(chunk, str):
            chunk = bytes(chunk, 'latin-1')
        hasher.update(chunk)
    computed = hasher.hexdigest()
    if computed != hash_value:
        raise IOError(errno.EPIPE, 'Corrupt image download. Hash was %s expected %s' % (computed, hash_value))