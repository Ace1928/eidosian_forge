import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError

        :param id: Server Group ID.
        :type id: ``str``
        :param name: Server Group Name.
        :type name: ``str``
        :param policy: Server Group policy.
        :type policy: ``str``
        :param members: Server Group members.
        :type members: ``list``
        :param rules: Server Group rules.
        :type rules: ``list``
        