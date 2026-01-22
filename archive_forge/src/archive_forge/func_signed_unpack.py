import base64
import hashlib
import hmac
import json
import os
import uuid
from oslo_utils import secretutils
from oslo_utils import uuidutils
def signed_unpack(data, hmac_data, hmac_keys):
    """Unpack data and check that it was signed with hmac_key.

    :param data: json string that was singed_packed.
    :param hmac_data: hmac data that was generated from json by hmac_key on
                      user side
    :param hmac_keys: server side hmac_keys, one of these should be the same
                      as user used to sign with

    :returns: None in case of something wrong, Object in case of everything OK.
    """
    if not (hmac_keys and hmac_data):
        return None
    hmac_data = hmac_data.strip()
    if not hmac_data:
        return None
    for hmac_key in hmac_keys:
        try:
            user_hmac_data = generate_hmac(data, hmac_key)
        except Exception:
            pass
        else:
            if secretutils.constant_time_compare(hmac_data, user_hmac_data):
                try:
                    contents = json.loads(binary_decode(base64.urlsafe_b64decode(data)))
                    contents['hmac_key'] = hmac_key
                    return contents
                except Exception:
                    return None
    return None