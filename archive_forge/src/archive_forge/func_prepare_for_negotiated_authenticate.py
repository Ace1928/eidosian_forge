import logging
import saml2
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2.client_base import Base
from saml2.client_base import LogoutError
from saml2.client_base import NoServiceDefined
from saml2.client_base import SignOnError
from saml2.httpbase import HTTPError
from saml2.ident import code
from saml2.ident import decode
from saml2.mdstore import locations
from saml2.s_utils import sid
from saml2.s_utils import status_message_factory
from saml2.s_utils import success_status_factory
from saml2.saml import AssertionIDRef
from saml2.samlp import STATUS_REQUEST_DENIED
from saml2.samlp import STATUS_UNKNOWN_PRINCIPAL
from saml2.time_util import not_on_or_after
def prepare_for_negotiated_authenticate(self, entityid=None, relay_state='', binding=None, vorg='', nameid_format=None, scoping=None, consent=None, extensions=None, sign=None, response_binding=saml2.BINDING_HTTP_POST, sigalg=None, digest_alg=None, **kwargs):
    """Makes all necessary preparations for an authentication request
        that negotiates which binding to use for authentication.

        :param entityid: The entity ID of the IdP to send the request to
        :param relay_state: To where the user should be returned after
            successfull log in.
        :param binding: Which binding to use for sending the request
        :param vorg: The entity_id of the virtual organization I'm a member of
        :param nameid_format:
        :param scoping: For which IdPs this query are aimed.
        :param consent: Whether the principal have given her consent
        :param extensions: Possible extensions
        :param sign: Whether the request should be signed or not.
        :param response_binding: Which binding to use for receiving the response
        :param kwargs: Extra key word arguments
        :return: session id and AuthnRequest info
        """
    expected_binding = binding
    bindings_to_try = [BINDING_HTTP_REDIRECT, BINDING_HTTP_POST] if not expected_binding else [expected_binding]
    binding_destinations = []
    unsupported_bindings = []
    for binding in bindings_to_try:
        try:
            destination = self._sso_location(entityid, binding)
        except Exception:
            unsupported_bindings.append(binding)
        else:
            binding_destinations.append((binding, destination))
    for binding, destination in binding_destinations:
        logger.debug('destination to provider: %s', destination)
        sign_redirect = sign and binding == BINDING_HTTP_REDIRECT
        sign_post = sign and (not sign_redirect)
        reqid, request = self.create_authn_request(destination=destination, vorg=vorg, scoping=scoping, binding=response_binding, nameid_format=nameid_format, consent=consent, extensions=extensions, sign=sign_post, sign_alg=sigalg, digest_alg=digest_alg, **kwargs)
        _req_str = str(request)
        logger.debug('AuthNReq: %s', _req_str)
        http_info = self.apply_binding(binding, _req_str, destination, relay_state, sign=sign_redirect, sigalg=sigalg)
        return (reqid, binding, http_info)
    else:
        error_context = {'message': 'No supported bindings available for authentication', 'bindings_to_try': bindings_to_try, 'unsupported_bindings': unsupported_bindings}
        raise SignOnError(error_context)