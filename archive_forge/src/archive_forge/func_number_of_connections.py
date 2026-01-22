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
def number_of_connections(self):
    if self._values['number_of_connections'] is None:
        return None
    if 0 <= self._values['number_of_connections'] <= 65535:
        return self._values['number_of_connections']
    raise F5ModuleError("Valid 'number_of_connections' must be in range 0 - 65535.")