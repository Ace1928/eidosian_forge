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
def create_authn_request(self, destination, vorg='', scoping=None, binding=BINDING_HTTP_POST, nameid_format=None, service_url_binding=None, message_id=0, consent=None, extensions=None, sign=None, sign_prepare=None, sign_alg=None, digest_alg=None, allow_create=None, requested_attributes=None, **kwargs):
    """Creates an authentication request.

        :param destination: Where the request should be sent.
        :param vorg: The virtual organization the service belongs to.
        :param scoping: The scope of the request
        :param binding: The protocol to use for the Response !!
        :param nameid_format: Format of the NameIDPolicy
        :param service_url_binding: Where the reply should be sent dependent
            on reply binding.
        :param message_id: The identifier for this request
        :param consent: Whether the principal have given her consent
        :param extensions: Possible extensions
        :param sign: Whether the request should be signed or not.
        :param sign_prepare: Whether the signature should be prepared or not.
        :param sign_alg: The request signature algorithm
        :param digest_alg: The request digest algorithm
        :param allow_create: If the identity provider is allowed, in the course
            of fulfilling the request, to create a new identifier to represent
            the principal.
        :param requested_attributes: A list of dicts which define attributes to
            be used as eIDAS Requested Attributes for this request. If not
            defined the configuration option requested_attributes will be used,
            if defined. The format is the same as the requested_attributes
            configuration option.
        :param kwargs: Extra key word arguments
        :return: either a tuple of request ID and <samlp:AuthnRequest> instance
                 or a tuple of request ID and str when sign is set to True
        """
    args = {}
    hide_assertion_consumer_service = self.config.getattr('hide_assertion_consumer_service', 'sp')
    assertion_consumer_service_url = kwargs.pop('assertion_consumer_service_urls', [None])[0] or kwargs.pop('assertion_consumer_service_url', None)
    assertion_consumer_service_index = kwargs.pop('assertion_consumer_service_index', None)
    service_url = (self.service_urls(service_url_binding or binding) or [None])[0]
    if hide_assertion_consumer_service:
        args['assertion_consumer_service_url'] = None
        binding = None
    elif assertion_consumer_service_url:
        args['assertion_consumer_service_url'] = assertion_consumer_service_url
    elif assertion_consumer_service_index:
        args['assertion_consumer_service_index'] = assertion_consumer_service_index
    elif service_url:
        args['assertion_consumer_service_url'] = service_url
    provider_name = kwargs.get('provider_name')
    if not provider_name and binding != BINDING_PAOS:
        provider_name = self._my_name()
    args['provider_name'] = provider_name
    requested_authn_context = kwargs.pop('requested_authn_context', None) or self.config.getattr('requested_authn_context', 'sp') or {}
    if isinstance(requested_authn_context, RequestedAuthnContext):
        args['requested_authn_context'] = requested_authn_context
    elif isinstance(requested_authn_context, Mapping):
        requested_authn_context_accrs = requested_authn_context.get('authn_context_class_ref', [])
        requested_authn_context_comparison = requested_authn_context.get('comparison', 'exact')
        if requested_authn_context_accrs:
            args['requested_authn_context'] = RequestedAuthnContext(authn_context_class_ref=[AuthnContextClassRef(accr) for accr in requested_authn_context_accrs], comparison=requested_authn_context_comparison)
    else:
        logger.warning({'message': 'Cannot process requested_authn_context', 'requested_authn_context': requested_authn_context, 'type_of_requested_authn_context': type(requested_authn_context)})
    _msg = AuthnRequest()
    for param in ['scoping', 'conditions', 'subject']:
        _item = kwargs.pop(param, None)
        if not _item:
            continue
        if isinstance(_item, _msg.child_class(param)):
            args[param] = _item
        else:
            raise ValueError(f'Wrong type for param {param}')
    nameid_policy_format_config = self.config.getattr('name_id_policy_format', 'sp')
    nameid_policy_format = nameid_format or nameid_policy_format_config or None
    allow_create_config = self.config.getattr('name_id_format_allow_create', 'sp')
    allow_create = None if nameid_policy_format == NAMEID_FORMAT_TRANSIENT else allow_create if allow_create else str(bool(allow_create_config)).lower()
    name_id_policy = kwargs.pop('name_id_policy', None) if 'name_id_policy' in kwargs else None if not nameid_policy_format else samlp.NameIDPolicy(allow_create=allow_create, format=nameid_policy_format)
    if name_id_policy and vorg:
        name_id_policy.sp_name_qualifier = vorg
        name_id_policy.format = nameid_policy_format or NAMEID_FORMAT_PERSISTENT
    args['name_id_policy'] = name_id_policy
    conf_sp_type = self.config.getattr('sp_type', 'sp')
    conf_sp_type_in_md = self.config.getattr('sp_type_in_metadata', 'sp')
    if conf_sp_type and conf_sp_type_in_md is False:
        if not extensions:
            extensions = Extensions()
        item = sp_type.SPType(text=conf_sp_type)
        extensions.add_extension_element(item)
    requested_attrs = requested_attributes or self.config.getattr('requested_attributes', 'sp') or []
    if requested_attrs:
        req_attrs_node = create_requested_attribute_node(requested_attrs, self.config.attribute_converters)
        if not extensions:
            extensions = Extensions()
        extensions.add_extension_element(req_attrs_node)
    force_authn = str(kwargs.pop('force_authn', None) or self.config.getattr('force_authn', 'sp')).lower() in ['true', '1']
    if force_authn:
        kwargs['force_authn'] = 'true'
    if kwargs:
        _args, extensions = self._filter_args(AuthnRequest(), extensions, **kwargs)
        args.update(_args)
    args.pop('id', None)
    nsprefix = kwargs.get('nsprefix')
    msg = self._message(AuthnRequest, destination, message_id, consent, extensions, sign, sign_prepare, sign_alg=sign_alg, digest_alg=digest_alg, protocol_binding=binding, scoping=scoping, nsprefix=nsprefix, **args)
    return msg