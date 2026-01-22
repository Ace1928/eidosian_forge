import calendar
import copy
import http.cookiejar as http_cookiejar
from http.cookies import SimpleCookie
import logging
import re
import time
from urllib.parse import urlencode
from urllib.parse import urlparse
import requests
from saml2 import SAMLError
from saml2 import class_name
from saml2.pack import make_soap_enveloped_saml_thingy
from saml2.time_util import utc_now
@staticmethod
def use_http_uri(message, typ, destination='', relay_state=''):
    if '\n' in message:
        data = message.split('\n')[1]
    else:
        data = message.strip()
    if typ == 'SAMLResponse':
        info = {'data': data, 'headers': [('Content-Type', 'application/samlassertion+xml'), ('Cache-Control', 'no-cache, no-store'), ('Pragma', 'no-cache')]}
    elif typ == 'SAMLRequest':
        if relay_state:
            query = urlencode({'ID': message, 'RelayState': relay_state})
        else:
            query = urlencode({'ID': message})
        info = {'data': '', 'url': f'{destination}?{query}'}
    else:
        raise NotImplementedError
    return info