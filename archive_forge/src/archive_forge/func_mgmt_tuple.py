from __future__ import absolute_import, division, print_function
import time
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def mgmt_tuple(self):
    Destination = namedtuple('Destination', ['ip', 'subnet'])
    try:
        parts = self._values['mgmt_address'].split('/')
        if len(parts) == 2:
            result = Destination(ip=parts[0], subnet=parts[1])
        elif len(parts) < 2:
            result = Destination(ip=parts[0], subnet=None)
        else:
            raise F5ModuleError('The provided mgmt_address is malformed.')
    except ValueError:
        result = Destination(ip=None, subnet=None)
    return result