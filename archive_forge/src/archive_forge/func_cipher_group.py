from __future__ import absolute_import, division, print_function
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def cipher_group(self):
    if self.want.cipher_group is None:
        return None
    if self.want.cipher_group == 'none' and self.have.cipher_group == 'none':
        return None
    if self.want.cipher_group != self.have.cipher_group:
        return self.want.cipher_group