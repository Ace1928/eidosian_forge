from __future__ import absolute_import, unicode_literals
import json
import os
import time
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from oauthlib.signals import scope_changed
from .errors import (InsecureTransportError, MismatchingStateError,
from .tokens import OAuth2Token
from .utils import is_secure_transport, list_to_scope, scope_to_list
def parse_implicit_response(uri, state=None, scope=None):
    """Parse the implicit token response URI into a dict.

    If the resource owner grants the access request, the authorization
    server issues an access token and delivers it to the client by adding
    the following parameters to the fragment component of the redirection
    URI using the ``application/x-www-form-urlencoded`` format:

    **access_token**
            REQUIRED.  The access token issued by the authorization server.

    **token_type**
            REQUIRED.  The type of the token issued as described in
            Section 7.1.  Value is case insensitive.

    **expires_in**
            RECOMMENDED.  The lifetime in seconds of the access token.  For
            example, the value "3600" denotes that the access token will
            expire in one hour from the time the response was generated.
            If omitted, the authorization server SHOULD provide the
            expiration time via other means or document the default value.

    **scope**
            OPTIONAL, if identical to the scope requested by the client,
            otherwise REQUIRED.  The scope of the access token as described
            by Section 3.3.

    **state**
            REQUIRED if the "state" parameter was present in the client
            authorization request.  The exact value received from the
            client.

    :param uri:
    :param state:
    :param scope:

    Similar to the authorization code response, but with a full token provided
    in the URL fragment:

    .. code-block:: http

        HTTP/1.1 302 Found
        Location: http://example.com/cb#access_token=2YotnFZFEjr1zCsicMWpAA
                &state=xyz&token_type=example&expires_in=3600
    """
    if not is_secure_transport(uri):
        raise InsecureTransportError()
    fragment = urlparse.urlparse(uri).fragment
    params = dict(urlparse.parse_qsl(fragment, keep_blank_values=True))
    for key in ('expires_in',):
        if key in params:
            params[key] = int(params[key])
    if 'scope' in params:
        params['scope'] = scope_to_list(params['scope'])
    if 'expires_in' in params:
        params['expires_at'] = time.time() + int(params['expires_in'])
    if state and params.get('state', None) != state:
        raise ValueError('Mismatching or missing state in params.')
    params = OAuth2Token(params, old_scope=scope)
    validate_token_parameters(params)
    return params