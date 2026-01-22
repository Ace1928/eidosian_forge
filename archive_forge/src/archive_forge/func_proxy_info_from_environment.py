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
def proxy_info_from_environment(method='http'):
    """Read proxy info from the environment variables.
    """
    if method not in ('http', 'https'):
        return
    env_var = method + '_proxy'
    url = os.environ.get(env_var, os.environ.get(env_var.upper()))
    if not url:
        return
    return proxy_info_from_url(url, method, noproxy=None)