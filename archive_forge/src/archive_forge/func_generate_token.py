import base64
import binascii
import hmac
import time
from oauth2client import _helpers
@_helpers.positional(2)
def generate_token(key, user_id, action_id='', when=None):
    """Generates a URL-safe token for the given user, action, time tuple.

    Args:
        key: secret key to use.
        user_id: the user ID of the authenticated user.
        action_id: a string identifier of the action they requested
                   authorization for.
        when: the time in seconds since the epoch at which the user was
              authorized for this action. If not set the current time is used.

    Returns:
        A string XSRF protection token.
    """
    digester = hmac.new(_helpers._to_bytes(key, encoding='utf-8'))
    digester.update(_helpers._to_bytes(str(user_id), encoding='utf-8'))
    digester.update(DELIMITER)
    digester.update(_helpers._to_bytes(action_id, encoding='utf-8'))
    digester.update(DELIMITER)
    when = _helpers._to_bytes(str(when or int(time.time())), encoding='utf-8')
    digester.update(when)
    digest = digester.digest()
    token = base64.urlsafe_b64encode(digest + DELIMITER + when)
    return token