from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def set_reasonable_creation_defaults(self):
    if self.want.action is None:
        self.changes.update({'action': 'reject'})
    if self.want.logging is None:
        self.changes.update({'logging': False})
    if self.want.status is None:
        self.changes.update({'status': 'enabled'})