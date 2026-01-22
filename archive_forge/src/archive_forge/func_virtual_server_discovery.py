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
def virtual_server_discovery(self):
    self._discovery_constraints()
    if self.want.virtual_server_discovery != self.have.virtual_server_discovery:
        return self.want.virtual_server_discovery