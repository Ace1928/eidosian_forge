import base64
import calendar
import copy
import email
import email.feedparser
from email import header
import email.message
import email.utils
import errno
from gettext import gettext as _
import gzip
from hashlib import md5 as _md5
from hashlib import sha1 as _sha
import hmac
import http.client
import io
import os
import random
import re
import socket
import ssl
import sys
import time
import urllib.parse
import zlib
from . import auth
from .error import *
from .iri2uri import iri2uri
from httplib2 import certs
def urlnorm(uri):
    scheme, authority, path, query, fragment = parse_uri(uri)
    if not scheme or not authority:
        raise RelativeURIError('Only absolute URIs are allowed. uri = %s' % uri)
    authority = authority.lower()
    scheme = scheme.lower()
    if not path:
        path = '/'
    request_uri = query and '?'.join([path, query]) or path
    scheme = scheme.lower()
    defrag_uri = scheme + '://' + authority + request_uri
    return (scheme, authority, request_uri, defrag_uri)