from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def lldp_tx_hold(self):
    if self._values['lldp'] is None:
        return None
    if self._values['lldp']['tx_hold'] is None:
        return None
    if 0 <= self._values['lldp']['tx_hold'] <= 65535:
        return self._values['lldp']['tx_hold']
    raise F5ModuleError("Valid 'tx_hold' must be in range 0 - 65535.")