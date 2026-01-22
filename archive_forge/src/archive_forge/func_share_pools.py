from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def share_pools(self):
    if self._values['idle_timeout_override'] is None:
        return None
    elif self._values['idle_timeout_override'] == 'enabled':
        return 'yes'
    elif self._values['idle_timeout_override'] == 'disabled':
        return 'no'