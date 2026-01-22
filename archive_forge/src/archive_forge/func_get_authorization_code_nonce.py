import logging
from oauthlib.oauth2.rfc6749.request_validator import (
def get_authorization_code_nonce(self, client_id, code, redirect_uri, request):
    """ Extracts nonce from saved authorization code.

        If present in the Authentication Request, Authorization
        Servers MUST include a nonce Claim in the ID Token with the
        Claim Value being the nonce value sent in the Authentication
        Request. Authorization Servers SHOULD perform no other
        processing on nonce values used. The nonce value is a
        case-sensitive string.

        Only code param should be sufficient to retrieve grant code from
        any storage you are using. However, `client_id` and `redirect_uri`
        have been validated and can be used also.

        :param client_id: Unicode client identifier
        :param code: Unicode authorization code grant
        :param redirect_uri: Unicode absolute URI
        :return: Unicode nonce

        Method is used by:
            - Authorization Token Grant Dispatcher
        """
    raise NotImplementedError('Subclasses must implement this method.')