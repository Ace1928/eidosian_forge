import logging
from oauthlib.oauth2.rfc6749.request_validator import (
def finalize_id_token(self, id_token, token, token_handler, request):
    """Finalize OpenID Connect ID token & Sign or Encrypt.

        In the OpenID Connect workflows when an ID Token is requested
        this method is called.  Subclasses should implement the
        construction, signing and optional encryption of the ID Token
        as described in the OpenID Connect spec.

        The `id_token` parameter is a dict containing a couple of OIDC
        technical fields related to the specification. Prepopulated
        attributes are:

        - `aud`, equals to `request.client_id`.
        - `iat`, equals to current time.
        - `nonce`, if present, is equals to the `nonce` from the
          authorization request.
        - `at_hash`, hash of `access_token`, if relevant.
        - `c_hash`, hash of `code`, if relevant.

        This method MUST provide required fields as below:

        - `iss`, REQUIRED. Issuer Identifier for the Issuer of the response.
        - `sub`, REQUIRED. Subject Identifier
        - `exp`, REQUIRED. Expiration time on or after which the ID
          Token MUST NOT be accepted by the RP when performing
          authentication with the OP.

        Additionals claims must be added, note that `request.scope`
        should be used to determine the list of claims.

        More information can be found at `OpenID Connect Core#Claims`_

        .. _`OpenID Connect Core#Claims`: https://openid.net/specs/openid-connect-core-1_0.html#Claims

        :param id_token: A dict containing technical fields of id_token
        :param token: A Bearer token dict
        :param token_handler: the token handler (BearerToken class)
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :return: The ID Token (a JWS signed JWT or JWE encrypted JWT)
        """
    raise NotImplementedError('Subclasses must implement this method.')