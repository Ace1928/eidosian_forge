from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def ip_ttl_v4(self):
    if self._values['ip_ttl_v4'] is None:
        return None
    if 0 <= self._values['ip_ttl_v4'] <= 255:
        return int(self._values['ip_ttl_v4'])
    raise F5ModuleError('ip_ttl_v4 must be between 0 and 255')