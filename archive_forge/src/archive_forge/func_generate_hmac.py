import base64
import hashlib
import hmac
import json
import os
import uuid
from oslo_utils import secretutils
from oslo_utils import uuidutils
def generate_hmac(data, hmac_key):
    """Generate a hmac using a known key given the provided content."""
    h = hmac.new(binary_encode(hmac_key), digestmod=hashlib.sha1)
    h.update(binary_encode(data))
    return h.hexdigest()