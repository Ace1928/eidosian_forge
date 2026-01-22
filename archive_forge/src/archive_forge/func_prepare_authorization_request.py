from __future__ import absolute_import, unicode_literals
import time
import warnings
from oauthlib.common import generate_token
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.errors import (InsecureTransportError,
from oauthlib.oauth2.rfc6749.parameters import (parse_token_response,
from oauthlib.oauth2.rfc6749.utils import is_secure_transport
def prepare_authorization_request(self, authorization_url, state=None, redirect_url=None, scope=None, **kwargs):
    """Prepare the authorization request.

        This is the first step in many OAuth flows in which the user is
        redirected to a certain authorization URL. This method adds
        required parameters to the authorization URL.

        :param authorization_url: Provider authorization endpoint URL.

        :param state: CSRF protection string. Will be automatically created if
        not provided. The generated state is available via the ``state``
        attribute. Clients should verify that the state is unchanged and
        present in the authorization response. This verification is done
        automatically if using the ``authorization_response`` parameter
        with ``prepare_token_request``.

        :param redirect_url: Redirect URL to which the user will be returned
        after authorization. Must be provided unless previously setup with
        the provider. If provided then it must also be provided in the
        token request.

        :param kwargs: Additional parameters to included in the request.

        :returns: The prepared request tuple with (url, headers, body).
        """
    if not is_secure_transport(authorization_url):
        raise InsecureTransportError()
    self.state = state or self.state_generator()
    self.redirect_url = redirect_url or self.redirect_url
    self.scope = scope or self.scope
    auth_url = self.prepare_request_uri(authorization_url, redirect_uri=self.redirect_url, scope=self.scope, state=self.state, **kwargs)
    return (auth_url, FORM_ENC_HEADERS, '')