from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def write_size(self):
    write = self._values['write_size']
    if write is None:
        return None
    if write < 2048 or write > 32768:
        raise F5ModuleError('Write Size value must be between 2048 and 32768')
    return self._values['write_size']