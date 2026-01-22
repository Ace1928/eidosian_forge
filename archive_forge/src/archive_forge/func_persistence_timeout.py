from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def persistence_timeout(self):
    if self._values['persistence_timeout'] is None:
        return None
    if 0 <= self._values['persistence_timeout'] <= 31536000:
        return self._values['persistence_timeout']
    raise F5ModuleError("Valid 'persistence_timeout' must be in range 0 - 31536000.")