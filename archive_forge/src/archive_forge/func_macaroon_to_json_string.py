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
def macaroon_to_json_string(macaroon):
    """Serialize macaroon object to a JSON-encoded string.

    @param macaroon object to be serialized.
    @return a string serialization form of the macaroon.
    """
    return macaroon.serialize(json_serializer.JsonSerializer())