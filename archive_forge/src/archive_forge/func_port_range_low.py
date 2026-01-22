from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def port_range_low(self):
    if self._values['port_range_low'] is None:
        return None
    high = self.port_range_high
    if 0 <= self._values['port_range_low'] <= 65535:
        if high:
            if high < self._values['port_range_low']:
                raise F5ModuleError("The 'port_range_low' value: {0} is lower than 'port_range_high' value: {1}".format(self._values['port_range_low'], high))
        return self._values['port_range_low']
    raise F5ModuleError("Valid 'port_range_low' must be in range 0 - 65535.")