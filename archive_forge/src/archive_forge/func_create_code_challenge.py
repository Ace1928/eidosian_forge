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
def create_code_challenge(self, code_verifier, code_challenge_method=None):
    """Create PKCE **code_challenge** derived from the  **code_verifier**.
        See `RFC7636 Section 4.2`_

        :param code_verifier: REQUIRED. The **code_verifier** generated from `create_code_verifier()`.
        :param code_challenge_method: OPTIONAL. The method used to derive the **code_challenge**. Acceptable values include `S256`. DEFAULT is `plain`.

               The client then creates a code challenge derived from the code
               verifier by using one of the following transformations on the code
               verifier::

                   plain
                      code_challenge = code_verifier
                   S256
                      code_challenge = BASE64URL-ENCODE(SHA256(ASCII(code_verifier)))

               If the client is capable of using `S256`, it MUST use `S256`, as
               `S256` is Mandatory To Implement (MTI) on the server.  Clients are
               permitted to use `plain` only if they cannot support `S256` for some
               technical reason and know via out-of-band configuration that the
               server supports `plain`.

               The plain transformation is for compatibility with existing
               deployments and for constrained environments that can't use the S256 transformation.

        .. _`RFC7636 Section 4.2`: https://tools.ietf.org/html/rfc7636#section-4.2
        """
    code_challenge = None
    if code_verifier == None:
        raise ValueError('Invalid code_verifier')
    if code_challenge_method == None:
        code_challenge_method = 'plain'
        self.code_challenge_method = code_challenge_method
        code_challenge = code_verifier
        self.code_challenge = code_challenge
    if code_challenge_method == 'S256':
        h = hashlib.sha256()
        h.update(code_verifier.encode(encoding='ascii'))
        sha256_val = h.digest()
        code_challenge = bytes.decode(base64.urlsafe_b64encode(sha256_val))
        code_challenge = code_challenge.replace('+', '-').replace('/', '_').replace('=', '')
        self.code_challenge = code_challenge
    return code_challenge