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
def parse_authn_request_response(self, xmlstr, binding, outstanding=None, outstanding_certs=None, conv_info=None):
    """Deal with an AuthnResponse

        :param xmlstr: The reply as a xml string
        :param binding: Which binding that was used for the transport
        :param outstanding: A dictionary with session IDs as keys and
            the original web request from the user before redirection
            as values.
        :param outstanding_certs:
        :param conv_info: Information about the conversation.
        :return: An response.AuthnResponse or None
        """
    if not getattr(self.config, 'entityid', None):
        raise SAMLError('Missing entity_id specification')
    if not xmlstr:
        return None
    kwargs = {'outstanding_queries': outstanding, 'outstanding_certs': outstanding_certs, 'allow_unsolicited': self.allow_unsolicited, 'want_assertions_signed': self.want_assertions_signed, 'want_assertions_or_response_signed': self.want_assertions_or_response_signed, 'want_response_signed': self.want_response_signed, 'return_addrs': self.service_urls(binding=binding), 'entity_id': self.config.entityid, 'attribute_converters': self.config.attribute_converters, 'allow_unknown_attributes': self.config.allow_unknown_attributes, 'conv_info': conv_info}
    try:
        resp = self._parse_response(xmlstr, AuthnResponse, 'assertion_consumer_service', binding, **kwargs)
    except StatusError as err:
        logger.error('SAML status error: %s', str(err))
        raise
    except UnravelError:
        return None
    except Exception as err:
        logger.error('XML parse error: %s', str(err))
        raise
    if not isinstance(resp, AuthnResponse):
        logger.error('Response type not supported: %s', saml2.class_name(resp))
        return None
    if resp.assertion and len(resp.response.encrypted_assertion) == 0 and resp.name_id:
        self.users.add_information_about_person(resp.session_info())
        logger.info('--- ADDED person info ----')
    return resp