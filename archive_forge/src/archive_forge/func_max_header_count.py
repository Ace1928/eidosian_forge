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
def max_header_count(self):
    if self._values['max_header_count'] is None:
        return None
    if self._values['max_header_count'] == 64:
        return 'default'
    return str(self._values['max_header_count'])