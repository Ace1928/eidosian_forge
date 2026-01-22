import base64
import hashlib
import hmac
import logging
import random
import string
import sys
import traceback
import zlib
from saml2 import VERSION
from saml2 import saml
from saml2 import samlp
from saml2.time_util import instant
def valid_email(emailaddress, domains=GENERIC_DOMAINS):
    """Checks for a syntactically valid email address."""
    if len(emailaddress) < 6:
        return False
    try:
        localpart, domainname = emailaddress.rsplit('@', 1)
        host, toplevel = domainname.rsplit('.', 1)
    except ValueError:
        return False
    if len(toplevel) != 2 and toplevel not in domains:
        return False
    for i in '-_.%+.':
        localpart = localpart.replace(i, '')
    for i in '-_.':
        host = host.replace(i, '')
    if localpart.isalnum() and host.isalnum():
        return True
    else:
        return False