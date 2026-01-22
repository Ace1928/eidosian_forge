import os
import json
import time
import base64
import requests
from libcloud.common.base import JsonResponse, ConnectionKey
def maybe_update_jwt(jwt):
    """
    Update jwt if it is expired

    :param jwt: jwt token to validate expiration
    :type  jwt: str

    :rtype: str
    """
    if is_jwt_expired(jwt):
        return refresh_jwt(jwt)
    return jwt