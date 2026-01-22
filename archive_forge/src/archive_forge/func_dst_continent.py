from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip_network
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def dst_continent(self):
    dst_continent = self._values['destination'].get('continent', None)
    if dst_continent is None:
        return None
    result = self.continents.get(dst_continent, dst_continent)
    return result