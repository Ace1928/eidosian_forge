import base64
import hashlib
import re
import secrets
import time
import warnings
from oauthlib.common import generate_token
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.errors import (
from oauthlib.oauth2.rfc6749.parameters import (
from oauthlib.oauth2.rfc6749.utils import is_secure_transport
def create_code_verifier(self, length):
    """Create PKCE **code_verifier** used in computing **code_challenge**. 
        See `RFC7636 Section 4.1`_

        :param length: REQUIRED. The length of the code_verifier.

        The client first creates a code verifier, "code_verifier", for each
        OAuth 2.0 [RFC6749] Authorization Request, in the following manner:

        .. code-block:: text

               code_verifier = high-entropy cryptographic random STRING using the
               unreserved characters [A-Z] / [a-z] / [0-9] / "-" / "." / "_" / "~"
               from Section 2.3 of [RFC3986], with a minimum length of 43 characters
               and a maximum length of 128 characters.

        .. _`RFC7636 Section 4.1`: https://tools.ietf.org/html/rfc7636#section-4.1
        """
    code_verifier = None
    if not length >= 43:
        raise ValueError('Length must be greater than or equal to 43')
    if not length <= 128:
        raise ValueError('Length must be less than or equal to 128')
    allowed_characters = re.compile('^[A-Zaa-z0-9-._~]')
    code_verifier = secrets.token_urlsafe(length)
    if not re.search(allowed_characters, code_verifier):
        raise ValueError('code_verifier contains invalid characters')
    self.code_verifier = code_verifier
    return code_verifier