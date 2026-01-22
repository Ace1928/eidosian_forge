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
def create_attribute_response(self, identity, in_response_to, destination, sp_entity_id, userid='', name_id=None, status=None, issuer=None, sign_assertion=None, sign_response=None, attributes=None, sign_alg=None, digest_alg=None, farg=None, **kwargs):
    """Create an attribute assertion response.

        :param identity: A dictionary with attributes and values that are
            expected to be the bases for the assertion in the response.
        :param in_response_to: The session identifier of the request
        :param destination: The URL which should receive the response
        :param sp_entity_id: The entity identifier of the SP
        :param userid: A identifier of the user
        :param name_id: The identifier of the subject
        :param status: The status of the response
        :param issuer: The issuer of the response
        :param sign_assertion: Whether the assertion should be signed or not
        :param sign_response: Whether the whole response should be signed
        :param attributes:
        :param kwargs: To catch extra keyword arguments
        :return: A response instance
        """
    policy = self.config.getattr('policy', 'aa')
    if not name_id and userid:
        try:
            name_id = self.ident.construct_nameid(userid, policy, sp_entity_id)
            logger.warning('Unspecified NameID format')
        except Exception:
            pass
    to_sign = []
    if identity:
        farg = self.update_farg(in_response_to, sp_entity_id, farg=farg)
        _issuer = self._issuer(issuer)
        ast = Assertion(identity)
        if policy:
            ast.apply_policy(sp_entity_id, policy)
        else:
            policy = Policy(mds=self.metadata)
        if attributes:
            restr = restriction_from_attribute_spec(attributes)
            ast = filter_attribute_value_assertions(ast, restr)
        assertion = ast.construct(sp_entity_id, self.config.attribute_converters, policy, issuer=_issuer, name_id=name_id, farg=farg['assertion'])
    return self._response(in_response_to, destination, status, issuer, sign_response, to_sign, sign_assertion=sign_assertion, sign_alg=sign_alg, digest_alg=digest_alg, assertion=assertion, sp_entity_id=sp_entity_id, **kwargs)