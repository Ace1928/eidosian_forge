import abc
import base64
import json
import logging
import os
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery._utils import b64decode
from pymacaroons.serializers import json_serializer
from ._versions import (
from ._error import (
from ._codec import (
from ._keys import PublicKey
from ._third_party import (
def macaroon_version(bakery_version):
    """Return the macaroon version given the bakery version.

    @param bakery_version the bakery version
    @return macaroon_version the derived macaroon version
    """
    if bakery_version in [VERSION_0, VERSION_1]:
        return pymacaroons.MACAROON_V1
    return pymacaroons.MACAROON_V2