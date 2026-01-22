from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def fqdns(self):
    if self.want.fqdns is None:
        return None
    elif self.have.fqdns is None:
        return self.want.fqdns
    if sorted(self.want.fqdns) != sorted(self.have.fqdns):
        return self.want.fqdns