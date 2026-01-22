from __future__ import absolute_import, unicode_literals
from oauthlib.common import Request, add_params_to_uri
from .. import errors
from .base import BaseEndpoint
def create_verifier(self, request, credentials):
    """Create and save a new request token.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :param credentials: A dict of extra token credentials.
        :returns: The verifier as a dict.
        """
    verifier = {'oauth_token': request.resource_owner_key, 'oauth_verifier': self.token_generator()}
    verifier.update(credentials)
    self.request_validator.save_verifier(request.resource_owner_key, verifier, request)
    return verifier