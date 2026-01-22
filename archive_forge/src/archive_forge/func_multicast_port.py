from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def multicast_port(self):
    if self._values['multicast_port'] is None:
        return None
    result = int(self._values['multicast_port'])
    if result < 0 or result > 65535:
        raise F5ModuleError("The specified 'multicast_port' must be between 0 and 65535.")
    return result