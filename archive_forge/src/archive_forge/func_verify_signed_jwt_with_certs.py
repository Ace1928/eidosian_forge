import json
import logging
import time
from oauth2client import _helpers
from oauth2client import _pure_python_crypt
def verify_signed_jwt_with_certs(jwt, certs, audience=None):
    """Verify a JWT against public certs.

    See http://self-issued.info/docs/draft-jones-json-web-token.html.

    Args:
        jwt: string, A JWT.
        certs: dict, Dictionary where values of public keys in PEM format.
        audience: string, The audience, 'aud', that this JWT should contain. If
                  None then the JWT's 'aud' parameter is not verified.

    Returns:
        dict, The deserialized JSON payload in the JWT.

    Raises:
        AppIdentityError: if any checks are failed.
    """
    jwt = _helpers._to_bytes(jwt)
    if jwt.count(b'.') != 2:
        raise AppIdentityError('Wrong number of segments in token: {0}'.format(jwt))
    header, payload, signature = jwt.split(b'.')
    message_to_sign = header + b'.' + payload
    signature = _helpers._urlsafe_b64decode(signature)
    payload_bytes = _helpers._urlsafe_b64decode(payload)
    try:
        payload_dict = json.loads(_helpers._from_bytes(payload_bytes))
    except:
        raise AppIdentityError("Can't parse token: {0}".format(payload_bytes))
    _verify_signature(message_to_sign, signature, certs.values())
    _verify_time_range(payload_dict)
    _check_audience(payload_dict, audience)
    return payload_dict