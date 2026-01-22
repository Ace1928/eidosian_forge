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
def maximum_age(self):
    if self._values['maximum_age'] is None:
        return None
    if self._values['maximum_age'] == 4294967295:
        return 'indefinite'
    return int(self._values['maximum_age'])