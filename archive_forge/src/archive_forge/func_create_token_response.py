from __future__ import absolute_import, unicode_literals
import json
import logging
from .. import errors
from ..request_validator import RequestValidator
from .base import GrantTypeBase
def create_token_response(self, request, token_handler):
    """Return token or error in json format.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :param token_handler: A token handler instance, for example of type
                              oauthlib.oauth2.BearerToken.

        If the access token request is valid and authorized, the
        authorization server issues an access token and optional refresh
        token as described in `Section 5.1`_.  If the request failed client
        authentication or is invalid, the authorization server returns an
        error response as described in `Section 5.2`_.

        .. _`Section 5.1`: https://tools.ietf.org/html/rfc6749#section-5.1
        .. _`Section 5.2`: https://tools.ietf.org/html/rfc6749#section-5.2
        """
    headers = self._get_default_headers()
    try:
        if self.request_validator.client_authentication_required(request):
            log.debug('Authenticating client, %r.', request)
            if not self.request_validator.authenticate_client(request):
                log.debug('Client authentication failed, %r.', request)
                raise errors.InvalidClientError(request=request)
        elif not self.request_validator.authenticate_client_id(request.client_id, request):
            log.debug('Client authentication failed, %r.', request)
            raise errors.InvalidClientError(request=request)
        log.debug('Validating access token request, %r.', request)
        self.validate_token_request(request)
    except errors.OAuth2Error as e:
        log.debug('Client error in token request, %s.', e)
        headers.update(e.headers)
        return (headers, e.json, e.status_code)
    token = token_handler.create_token(request, self.refresh_token)
    for modifier in self._token_modifiers:
        token = modifier(token)
    self.request_validator.save_token(token, request)
    log.debug('Issuing token %r to client id %r (%r) and username %s.', token, request.client_id, request.client, request.username)
    return (headers, json.dumps(token), 200)