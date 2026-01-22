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
def persistent_nameid(self, userid, sp_name_qualifier='', name_qualifier=''):
    nameid = self.match_local_id(userid, sp_name_qualifier, name_qualifier)
    if nameid:
        return nameid
    else:
        return self.get_nameid(userid, NAMEID_FORMAT_PERSISTENT, sp_name_qualifier, name_qualifier)