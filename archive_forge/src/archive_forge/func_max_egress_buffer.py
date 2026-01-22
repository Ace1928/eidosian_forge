from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def max_egress_buffer(self):
    if self._values['max_egress_buffer'] is None:
        return None
    if 0 <= self._values['max_egress_buffer'] <= 4294967295:
        return self._values['max_egress_buffer']
    raise F5ModuleError("Valid 'max_egress_buffer' must be in range 0 - 4294967295.")