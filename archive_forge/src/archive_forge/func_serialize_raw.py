from __future__ import unicode_literals
import binascii
from collections import namedtuple
import six
import struct
import sys
from base64 import urlsafe_b64encode
from pymacaroons.utils import (
from pymacaroons.serializers.base_serializer import BaseSerializer
from pymacaroons.exceptions import MacaroonSerializationException
def serialize_raw(self, macaroon):
    from pymacaroons.macaroon import MACAROON_V1
    if macaroon.version == MACAROON_V1:
        return self._serialize_v1(macaroon)
    return self._serialize_v2(macaroon)