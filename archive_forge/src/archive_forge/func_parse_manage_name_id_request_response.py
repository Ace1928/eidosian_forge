import base64
from binascii import hexlify
import copy
from hashlib import sha1
import logging
import zlib
import requests
from saml2 import BINDING_HTTP_ARTIFACT
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_PAOS
from saml2 import BINDING_SOAP
from saml2 import BINDING_URI
from saml2 import VERSION
from saml2 import SamlBase
from saml2 import SAMLError
from saml2 import class_name
from saml2 import element_to_extension_element
from saml2 import extension_elements_to_elements
from saml2 import request as saml_request
from saml2 import response as saml_response
from saml2 import saml
from saml2 import samlp
from saml2 import soap
from saml2.config import config_factory
from saml2.httpbase import HTTPBase
from saml2.mdstore import all_locations
from saml2.metadata import ENDPOINTS
from saml2.pack import http_form_post_message
from saml2.pack import http_redirect_message
from saml2.profile import ecp
from saml2.profile import paos
from saml2.profile import samlec
from saml2.response import LogoutResponse
from saml2.response import UnsolicitedResponse
from saml2.s_utils import UnravelError
from saml2.s_utils import UnsupportedBinding
from saml2.s_utils import decode_base64_and_inflate
from saml2.s_utils import error_status_factory
from saml2.s_utils import rndbytes
from saml2.s_utils import sid
from saml2.s_utils import success_status_factory
from saml2.saml import NAMEID_FORMAT_ENTITY
from saml2.saml import EncryptedAssertion
from saml2.saml import Issuer
from saml2.saml import NameID
from saml2.samlp import Artifact
from saml2.samlp import ArtifactResolve
from saml2.samlp import ArtifactResponse
from saml2.samlp import AssertionIDRequest
from saml2.samlp import AttributeQuery
from saml2.samlp import AuthnQuery
from saml2.samlp import AuthnRequest
from saml2.samlp import AuthzDecisionQuery
from saml2.samlp import LogoutRequest
from saml2.samlp import ManageNameIDRequest
from saml2.samlp import NameIDMappingRequest
from saml2.samlp import SessionIndex
from saml2.samlp import artifact_resolve_from_string
from saml2.samlp import response_from_string
from saml2.sigver import SignatureError
from saml2.sigver import SigverError
from saml2.sigver import get_pem_wrapped_unwrapped
from saml2.sigver import make_temp
from saml2.sigver import pre_encrypt_assertion
from saml2.sigver import pre_encryption_part
from saml2.sigver import pre_signature_part
from saml2.sigver import security_context
from saml2.sigver import signed_instance_factory
from saml2.soap import class_instances_from_soap_enveloped_saml_thingies
from saml2.soap import open_soap_envelope
from saml2.soap import parse_soap_enveloped_saml_artifact_resolve
from saml2.time_util import instant
from saml2.virtual_org import VirtualOrg
from saml2.xmldsig import DIGEST_ALLOWED_ALG
from saml2.xmldsig import SIG_ALLOWED_ALG
from saml2.xmldsig import DefaultSignature
def parse_manage_name_id_request_response(self, string, binding=BINDING_SOAP):
    return self._parse_response(string, saml_response.ManageNameIDResponse, 'manage_name_id_service', binding, asynchop=False)