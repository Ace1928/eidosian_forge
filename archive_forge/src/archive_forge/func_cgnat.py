from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def cgnat(self):
    if self.want.module == 'cgnat':
        if self.want.state == 'absent' and self.have.enabled is True:
            return True
        if self.want.state == 'present' and self.have.disabled is True:
            return True