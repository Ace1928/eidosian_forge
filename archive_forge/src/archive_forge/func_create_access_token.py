from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import urlencode
from .. import errors
from .base import BaseEndpoint
def create_access_token(self, request, credentials):
    """Create and save a new access token.

        Similar to OAuth 2, indication of granted scopes will be included as a
        space separated list in ``oauth_authorized_realms``.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: The token as an urlencoded string.
        """
    request.realms = self.request_validator.get_realms(request.resource_owner_key, request)
    token = {'oauth_token': self.token_generator(), 'oauth_token_secret': self.token_generator(), 'oauth_authorized_realms': ' '.join(request.realms)}
    token.update(credentials)
    self.request_validator.save_access_token(token, request)
    return urlencode(token.items())