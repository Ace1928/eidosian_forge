from __future__ import absolute_import, unicode_literals
import base64
import hashlib
import json
import logging
from oauthlib import common
from .. import errors
from .base import GrantTypeBase
def code_challenge_method_s256(verifier, challenge):
    """
    If the "code_challenge_method" from `Section 4.3`_ was "S256", the
    received "code_verifier" is hashed by SHA-256, base64url-encoded, and
    then compared to the "code_challenge", i.e.:

    BASE64URL-ENCODE(SHA256(ASCII(code_verifier))) == code_challenge

    How to implement a base64url-encoding
    function without padding, based upon the standard base64-encoding
    function that uses padding.

    To be concrete, example C# code implementing these functions is shown
    below.  Similar code could be used in other languages.

    static string base64urlencode(byte [] arg)
    {
        string s = Convert.ToBase64String(arg); // Regular base64 encoder
        s = s.Split('=')[0]; // Remove any trailing '='s
        s = s.Replace('+', '-'); // 62nd char of encoding
        s = s.Replace('/', '_'); // 63rd char of encoding
        return s;
    }

    In python urlsafe_b64encode is already replacing '+' and '/', but preserve
    the trailing '='. So we have to remove it.

    .. _`Section 4.3`: https://tools.ietf.org/html/rfc7636#section-4.3
    """
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=') == challenge