from __future__ import absolute_import, division, print_function
import copy
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def fallback_ip(self):
    if self._values['fallback_ip'] is None:
        return None
    if self._values['fallback_ip'] == 'any':
        return 'any'
    if self._values['fallback_ip'] == 'any6':
        return 'any6'
    if is_valid_ip(self._values['fallback_ip']):
        return self._values['fallback_ip']
    else:
        raise F5ModuleError('The provided fallback address is not a valid IPv4 address')