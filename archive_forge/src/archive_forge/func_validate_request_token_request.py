from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import urlencode
from .. import errors
from .base import BaseEndpoint
def validate_request_token_request(self, request):
    """Validate a request token request.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :raises: OAuth1Error if the request is invalid.
        :returns: A tuple of 2 elements.
                  1. The validation result (True or False).
                  2. The request object.
        """
    self._check_transport_security(request)
    self._check_mandatory_parameters(request)
    if request.realm:
        request.realms = request.realm.split(' ')
    else:
        request.realms = self.request_validator.get_default_realms(request.client_key, request)
    if not self.request_validator.check_realms(request.realms):
        raise errors.InvalidRequestError(description='Invalid realm %s. Allowed are %r.' % (request.realms, self.request_validator.realms))
    if not request.redirect_uri:
        raise errors.InvalidRequestError(description='Missing callback URI.')
    if not self.request_validator.validate_timestamp_and_nonce(request.client_key, request.timestamp, request.nonce, request, request_token=request.resource_owner_key):
        return (False, request)
    valid_client = self.request_validator.validate_client_key(request.client_key, request)
    if not valid_client:
        request.client_key = self.request_validator.dummy_client
    valid_realm = self.request_validator.validate_requested_realms(request.client_key, request.realms, request)
    valid_redirect = self.request_validator.validate_redirect_uri(request.client_key, request.redirect_uri, request)
    if not request.redirect_uri:
        raise NotImplementedError('Redirect URI must either be provided or set to a default during validation.')
    valid_signature = self._check_signature(request)
    request.validator_log['client'] = valid_client
    request.validator_log['realm'] = valid_realm
    request.validator_log['callback'] = valid_redirect
    request.validator_log['signature'] = valid_signature
    v = all((valid_client, valid_realm, valid_redirect, valid_signature))
    if not v:
        log.info('[Failure] request verification failed.')
        log.info('Valid client: %s.', valid_client)
        log.info('Valid realm: %s.', valid_realm)
        log.info('Valid callback: %s.', valid_redirect)
        log.info('Valid signature: %s.', valid_signature)
    return (v, request)