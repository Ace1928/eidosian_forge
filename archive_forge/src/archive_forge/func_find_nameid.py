import datetime
from hashlib import sha1
import logging
from pymongo import MongoClient
import pymongo.errors
import pymongo.uri_parser
from saml2.eptid import Eptid
from saml2.ident import IdentDB
from saml2.ident import Unknown
from saml2.ident import code_binary
from saml2.mdie import from_dict
from saml2.mdie import to_dict
from saml2.mdstore import InMemoryMetaData
from saml2.mdstore import load_metadata_modules
from saml2.mdstore import metadata_modules
from saml2.s_utils import PolicyError
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def find_nameid(self, userid, nformat=None, sp_name_qualifier=None, name_qualifier=None, sp_provided_id=None, **kwargs):
    kwargs = {}
    if nformat:
        kwargs['name_format'] = nformat
    if sp_name_qualifier:
        kwargs['sp_name_qualifier'] = sp_name_qualifier
    if name_qualifier:
        kwargs['name_qualifier'] = name_qualifier
    if sp_provided_id:
        kwargs['sp_provided_id'] = sp_provided_id
    res = []
    for item in self.mdb.get(userid, **kwargs):
        res.append(from_dict(item['name_id'], ONTS, True))
    return res