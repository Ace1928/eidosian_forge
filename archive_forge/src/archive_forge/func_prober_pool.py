from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def prober_pool(self):
    if self.want.prober_pool is None:
        return None
    if self.have.prober_pool is None:
        if self.want.prober_pool == '':
            return None
    if self.want.prober_pool != self.have.prober_pool:
        return self.want.prober_pool