from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_dictionary
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def ip_intelligence(self):
    if self._values['ip_intelligence'] is None:
        return None
    to_filter = dict(log_publisher=self._values['ip_log_publisher'], rate_limit=self._change_rate_limit_value(self._values['ip_rate_limit']), log_rtbh=self.ip_log_rtbh, log_shun=self.ip_log_shun, log_translation_fields=self.ip_log_translation_fields)
    result = self._filter_params(to_filter)
    if result:
        return result