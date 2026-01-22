from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def tsig_key(self):
    if self.want.tsig_key is None:
        return None
    if self.have.tsig_key is None and self.want.tsig_key == '':
        return None
    if self.want.tsig_key != self.have.tsig_key:
        return self.want.tsig_key