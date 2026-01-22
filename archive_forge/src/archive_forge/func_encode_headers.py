import copy
import io
import logging
import socket
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as ksa_exc
import OpenSSL
from oslo_utils import importutils
from oslo_utils import netutils
import requests
import urllib.parse
from oslo_utils import encodeutils
from glanceclient.common import utils
from glanceclient import exc
def encode_headers(headers):
    """Encodes headers.

    Note: This should be used right before
    sending anything out.

    :param headers: Headers to encode
    :returns: Dictionary with encoded headers'
              names and values
    """
    encoded_dict = {}
    for h, v in headers.items():
        if v is not None:
            safe = '=+/' if h in TOKEN_HEADERS else '/:'
            key = urllib.parse.quote(h, safe)
            value = urllib.parse.quote(v, safe)
            encoded_dict[key] = value
    return dict(((encodeutils.safe_encode(h, encoding='ascii'), encodeutils.safe_encode(v, encoding='ascii')) for h, v in encoded_dict.items()))