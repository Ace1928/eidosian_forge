from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def translation_port_range(self):
    if self.want.port_range_low is None:
        return None
    if self.want.port_range_low != self.have.port_range_low:
        result = '{0}-{1}'.format(self.want.port_range_low, self.want.port_range_high)
        return result
    if self.want.port_range_high != self.have.port_range_high:
        result = '{0}-{1}'.format(self.want.port_range_low, self.want.port_range_high)
        return result
    return None