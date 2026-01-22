import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def lru_cache_weakref(*cache_args, **cache_kwargs):
    """
    Version of functools.lru_cache that stores a weak reference to ``self``.

    Serves the same purpose as :py:func:`instance_cache` but uses Python's
    functools implementation which offers ``max_size`` and ``typed`` properties.

    lru_cache is a global cache even when used on a method. The cache's
    reference to ``self`` will prevent garbace collection of the object. This
    wrapper around functools.lru_cache replaces the reference to ``self`` with
    a weak reference to not interfere with garbage collection.
    """

    def wrapper(func):

        @functools.lru_cache(*cache_args, **cache_kwargs)
        def func_with_weakref(weakref_to_self, *args, **kwargs):
            return func(weakref_to_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return func_with_weakref(weakref.ref(self), *args, **kwargs)
        inner.cache_info = func_with_weakref.cache_info
        return inner
    return wrapper