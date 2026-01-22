from __future__ import absolute_import, unicode_literals
import logging
from itertools import chain
from oauthlib.common import add_params_to_uri
from oauthlib.uri_validate import is_absolute_uri
from oauthlib.oauth2.rfc6749 import errors, utils
from ..request_validator import RequestValidator
def prepare_authorization_response(self, request, token, headers, body, status):
    """Place token according to response mode.

        Base classes can define a default response mode for their authorization
        response by overriding the static `default_response_mode` member.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :param token:
        :param headers:
        :param body:
        :param status:
        """
    request.response_mode = request.response_mode or self.default_response_mode
    if request.response_mode not in ('query', 'fragment'):
        log.debug('Overriding invalid response mode %s with %s', request.response_mode, self.default_response_mode)
        request.response_mode = self.default_response_mode
    token_items = token.items()
    if request.response_type == 'none':
        state = token.get('state', None)
        if state:
            token_items = [('state', state)]
        else:
            token_items = []
    if request.response_mode == 'query':
        headers['Location'] = add_params_to_uri(request.redirect_uri, token_items, fragment=False)
        return (headers, body, status)
    if request.response_mode == 'fragment':
        headers['Location'] = add_params_to_uri(request.redirect_uri, token_items, fragment=True)
        return (headers, body, status)
    raise NotImplementedError('Subclasses must set a valid default_response_mode')