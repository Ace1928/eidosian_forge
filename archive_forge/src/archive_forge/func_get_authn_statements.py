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
def get_authn_statements(self, name_id, session_index=None, requested_context=None):
    """

        :param name_id:
        :param session_index:
        :param requested_context:
        :return:
        """
    return [k.authn_statement for k in self.get_assertions_by_subject(name_id, session_index, requested_context)]