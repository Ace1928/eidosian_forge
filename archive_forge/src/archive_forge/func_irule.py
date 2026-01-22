from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def irule(self):
    if self.want.irule is None:
        return None
    if self.have.irule is None and self.want.irule == '':
        return None
    if self.have.irule is None:
        return self.want.irule
    if self.want.irule != self.have.irule:
        return self.want.irule