import cgi
import hashlib
import hmac
from http.cookies import SimpleCookie
import logging
import time
from typing import Optional
from urllib.parse import parse_qs
from urllib.parse import quote
from saml2 import BINDING_HTTP_ARTIFACT
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import BINDING_URI
from saml2 import SAMLError
from saml2 import time_util
def make_cookie(name, load, seed, expire=0, domain='', path='', timestamp=''):
    """
    Create and return a cookie

    :param name: Cookie name
    :param load: Cookie load
    :param seed: A seed for the HMAC function
    :param expire: Number of minutes before this cookie goes stale
    :param domain: The domain of the cookie
    :param path: The path specification for the cookie
    :return: A tuple to be added to headers
    """
    cookie = SimpleCookie()
    if not timestamp:
        timestamp = str(int(time.mktime(time.gmtime())))
    signature = cookie_signature(seed, load, timestamp)
    cookie[name] = '|'.join([load, timestamp, signature])
    if path:
        cookie[name]['path'] = path
    if domain:
        cookie[name]['domain'] = domain
    if expire:
        cookie[name]['expires'] = _expiration(expire, '%a, %d-%b-%Y %H:%M:%S GMT')
    return tuple(cookie.output().split(': ', 1))