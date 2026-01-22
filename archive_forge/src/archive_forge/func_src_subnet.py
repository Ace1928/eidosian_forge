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
def src_subnet(self):
    src_subnet = self._values['source'].get('subnet', None)
    if src_subnet is None:
        return None
    if is_valid_ip_network(src_subnet):
        return src_subnet
    raise F5ModuleError("Specified 'subnet' is not a valid subnet.")