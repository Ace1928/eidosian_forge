import copy
from hashlib import sha256
import logging
import shelve
from urllib.parse import quote
from urllib.parse import unquote
from saml2 import SAMLError
from saml2.s_utils import PolicyError
from saml2.s_utils import rndbytes
from saml2.saml import NAMEID_FORMAT_EMAILADDRESS
from saml2.saml import NAMEID_FORMAT_PERSISTENT
from saml2.saml import NAMEID_FORMAT_TRANSIENT
from saml2.saml import NameID
def code_binary(item):
    """
    Return a binary 'code' suitable for hashing.
    """
    code_str = code(item)
    if isinstance(code_str, str):
        return code_str.encode('utf-8')
    return code_str