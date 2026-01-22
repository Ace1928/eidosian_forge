from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def header_erase(self):
    if self.want.header_erase is None:
        return None
    if self.want.header_erase in ['none', '']:
        if self.have.header_erase in [None, 'none']:
            return None
    if self.want.header_erase != self.have.header_erase:
        return self.want.header_erase