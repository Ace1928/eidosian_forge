from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def default_vs_syn_challenge_tresh(self):
    if self._values['default_vs_syn_challenge_tresh'] is None:
        return None
    value = self._values['default_vs_syn_challenge_tresh']
    if value is None:
        return None
    if value == 'infinite':
        return 0
    return int(value)