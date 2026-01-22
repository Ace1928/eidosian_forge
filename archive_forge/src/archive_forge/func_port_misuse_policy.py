from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def port_misuse_policy(self):
    if self._values['port_misuse_policy'] is None:
        return None
    if self._values['port_misuse_policy'] == '':
        return ''
    return fq_name(self.partition, self._values['port_misuse_policy'])