from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def spanning(self):
    if self.want.spanning is None:
        return None
    if self.want.spanning != self.have.spanning:
        return self.want.spanning