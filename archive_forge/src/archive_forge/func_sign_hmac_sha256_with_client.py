from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def sign_hmac_sha256_with_client(base_string, client):
    return sign_hmac_sha256(base_string, client.client_secret, client.resource_owner_secret)