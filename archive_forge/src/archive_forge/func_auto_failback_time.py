from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def auto_failback_time(self):
    if self._values['auto_failback_time'] is None:
        return None
    value = self._values['auto_failback_time']
    if value < 0 or value > 300:
        raise F5ModuleError('Invalid auto_failback_time value, correct range is 0 - 300, specified value: {0}.'.format(value))
    return value