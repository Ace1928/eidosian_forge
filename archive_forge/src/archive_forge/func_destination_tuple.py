from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def destination_tuple(self):
    pattern = '^[a-zA-Z0-9_.-]+'
    Destination = namedtuple('Destination', ['ip', 'port', 'route_domain', 'mask', 'not_ip'])
    if self._values['destination'] is None:
        result = Destination(ip=None, port=None, route_domain=None, mask=None, not_ip=None)
        return result
    addr = self._values['destination'].split('%')[0].split('/')[0]
    if is_valid_ip(addr):
        addr = compress_address(u'{0}'.format(addr))
        result = Destination(ip=addr, port=self.port, route_domain=self.route_domain, mask=self.mask, not_ip=False)
        return result
    else:
        matches = re.search(pattern, addr)
        if matches:
            result = Destination(ip=addr, port=self.port, route_domain=self.route_domain, mask=self.mask, not_ip=True)
            return result
    result = Destination(ip=addr, port=self.port, route_domain=self.route_domain, mask=self.mask, not_ip=False)
    return result