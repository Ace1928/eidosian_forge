import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider

        Set plain_auth as an extra :class:`OpenNebulaConnection_3_8` argument

        :return: ``dict`` of :class:`OpenNebulaConnection_3_8` input arguments
        