import json
import os
import six
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import jwt
import google.auth.transport.requests
def verify_firebase_token(id_token, request, audience=None, clock_skew_in_seconds=0):
    """Verifies an ID Token issued by Firebase Authentication.

    Args:
        id_token (Union[str, bytes]): The encoded token.
        request (google.auth.transport.Request): The object used to make
            HTTP requests.
        audience (str): The audience that this token is intended for. This is
            typically your Firebase application ID. If None then the audience
            is not verified.
        clock_skew_in_seconds (int): The clock skew used for `iat` and `exp`
            validation.

    Returns:
        Mapping[str, Any]: The decoded token.
    """
    return verify_token(id_token, request, audience=audience, certs_url=_GOOGLE_APIS_CERTS_URL, clock_skew_in_seconds=clock_skew_in_seconds)