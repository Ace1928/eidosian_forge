import logging
import threading
import time
from typing import Mapping
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlparse
from warnings import warn as _warn
import saml2
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_PAOS
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2 import samlp
from saml2 import soap
from saml2.entity import Entity
from saml2.extension import sp_type
from saml2.extension.requested_attributes import RequestedAttribute
from saml2.extension.requested_attributes import RequestedAttributes
from saml2.mdstore import locations
from saml2.population import Population
from saml2.profile import ecp
from saml2.profile import paos
from saml2.response import AssertionIDResponse
from saml2.response import AttributeResponse
from saml2.response import AuthnQueryResponse
from saml2.response import AuthnResponse
from saml2.response import AuthzResponse
from saml2.response import NameIDMappingResponse
from saml2.response import StatusError
from saml2.s_utils import UnravelError
from saml2.s_utils import do_attributes
from saml2.s_utils import signature
from saml2.saml import NAMEID_FORMAT_PERSISTENT
from saml2.saml import NAMEID_FORMAT_TRANSIENT
from saml2.saml import AuthnContextClassRef
from saml2.samlp import AttributeQuery
from saml2.samlp import AuthnQuery
from saml2.samlp import AuthnRequest
from saml2.samlp import AuthzDecisionQuery
from saml2.samlp import Extensions
from saml2.samlp import NameIDMappingRequest
from saml2.samlp import RequestedAuthnContext
from saml2.soap import make_soap_enveloped_saml_thingy
@staticmethod
def create_discovery_service_request(url, entity_id, **kwargs):
    """
        Created the HTTP redirect URL needed to send the user to the
        discovery service.

        :param url: The URL of the discovery service
        :param entity_id: The unique identifier of the service provider
        :param return: The discovery service MUST redirect the user agent
            to this location in response to this request
        :param policy: A parameter name used to indicate the desired behavior
            controlling the processing of the discovery service
        :param returnIDParam: A parameter name used to return the unique
            identifier of the selected identity provider to the original
            requester.
        :param isPassive: A boolean value True/False that controls
            whether the discovery service is allowed to visibly interact with
            the user agent.
        :return: A URL
        """
    args = {'entityID': entity_id, 'policy': kwargs.get('policy'), 'returnIDParam': kwargs.get('returnIDParam'), 'return': kwargs.get('return_url') or kwargs.get('return'), 'isPassive': None if 'isPassive' not in kwargs.keys() else 'true' if kwargs.get('isPassive') else 'false'}
    params = urlencode({k: v for k, v in args.items() if v})
    if '?' in url:
        return f'{url}&{params}'
    else:
        return f'{url}?{params}'