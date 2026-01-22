import ssl
import time
import socket
import logging
from datetime import datetime, timedelta
from functools import wraps
from libcloud.utils.py3 import httplib
from libcloud.common.exceptions import RateLimitReachedError
def transform_ssl_error(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ssl.SSLError as exc:
        if TRANSIENT_SSL_ERROR in str(exc):
            raise TransientSSLError(*exc.args)
        raise exc