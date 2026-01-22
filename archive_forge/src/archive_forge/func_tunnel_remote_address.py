from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def tunnel_remote_address(self):
    if self._values['tunnel_remote_address'] is None:
        return None
    if self._values['route_domain'] and len(self._values['tunnel_remote_address'].split('%')) == 1:
        result = '{0}%{1}'.format(self._values['tunnel_remote_address'], self._values['route_domain'])
        return result
    return self._values['tunnel_remote_address']