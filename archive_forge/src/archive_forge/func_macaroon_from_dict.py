import base64
import binascii
import ipaddress
import json
import webbrowser
from datetime import datetime
import six
from pymacaroons import Macaroon
from pymacaroons.serializers import json_serializer
import six.moves.http_cookiejar as http_cookiejar
from six.moves.urllib.parse import urlparse
def macaroon_from_dict(json_macaroon):
    """Return a pymacaroons.Macaroon object from the given
    JSON-deserialized dict.

    @param JSON-encoded macaroon as dict
    @return the deserialized macaroon object.
    """
    return Macaroon.deserialize(json.dumps(json_macaroon), json_serializer.JsonSerializer())