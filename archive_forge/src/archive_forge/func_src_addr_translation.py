from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def src_addr_translation(self):
    if self._values['snat_type'] is None:
        return None
    to_filter = dict(pool=self._values['snat_pool'], type=self._values['snat_type'])
    result = self._filter_params(to_filter)
    if result:
        return result