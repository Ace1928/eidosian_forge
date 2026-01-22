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
def get_assertions_by_subject(self, name_id=None, session_index=None, requested_context=None):
    """

        :param name_id: One of name_id or key can be used to get the authn
            statement
        :param session_index: If match against a session index should be done
        :param requested_context: Authn statements should match a specific
            authn context
        :return:
        """
    result = []
    key = sha1(code_binary(name_id)).hexdigest()
    for item in self.assertion.find({'name_id_key': key}):
        assertion = from_dict(item['assertion'], ONTS, True)
        if session_index or requested_context:
            for statement in assertion.authn_statement:
                if session_index:
                    if statement.session_index == session_index:
                        result.append(assertion)
                        break
                if requested_context:
                    if context_match(requested_context, statement.authn_context):
                        result.append(assertion)
                        break
        else:
            result.append(assertion)
    return result