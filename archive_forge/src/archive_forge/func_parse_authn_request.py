import dbm
import importlib
import logging
import shelve
import threading
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import class_name
from saml2 import element_to_extension_element
from saml2 import saml
from saml2.argtree import add_path
from saml2.argtree import is_set
from saml2.assertion import Assertion
from saml2.assertion import Policy
from saml2.assertion import filter_attribute_value_assertions
from saml2.assertion import restriction_from_attribute_spec
import saml2.cryptography.symmetric
from saml2.entity import Entity
from saml2.eptid import Eptid
from saml2.eptid import EptidShelve
from saml2.ident import IdentDB
from saml2.ident import decode
from saml2.profile import ecp
from saml2.request import AssertionIDRequest
from saml2.request import AttributeQuery
from saml2.request import AuthnQuery
from saml2.request import AuthnRequest
from saml2.request import AuthzDecisionQuery
from saml2.request import NameIDMappingRequest
from saml2.s_utils import MissingValue
from saml2.s_utils import Unknown
from saml2.s_utils import rndstr
from saml2.samlp import NameIDMappingResponse
from saml2.schema import soapenv
from saml2.sdb import SessionStorage
from saml2.sigver import CertificateError
from saml2.sigver import pre_signature_part
from saml2.sigver import signed_instance_factory
def parse_authn_request(self, enc_request, binding=BINDING_HTTP_REDIRECT, relay_state=None, sigalg=None, signature=None):
    """Parse a Authentication Request

        :param enc_request: The request in its transport format
        :param binding: Which binding that was used to transport the message
        :param relay_state: RelayState, when binding=redirect
        :param sigalg: Signature Algorithm, when binding=redirect
        :param signature: Signature, when binding=redirect
            to this entity.
        :return: A request instance
        """
    return self._parse_request(enc_request, AuthnRequest, 'single_sign_on_service', binding, relay_state=relay_state, sigalg=sigalg, signature=signature)