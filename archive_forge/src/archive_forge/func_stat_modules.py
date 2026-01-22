from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def stat_modules(self):
    if self._values['statistics'] is None:
        return None
    modules = self._values['statistics']['stat_modules']
    result = list()
    for module in modules:
        result.append(dict(module=module.upper()))
    return result