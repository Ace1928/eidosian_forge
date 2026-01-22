from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def latency_threshold(self):
    if self._values['heavy_urls'] is None or self._values['heavy_urls']['latency_threshold'] is None:
        return None
    if 0 <= self._values['heavy_urls']['latency_threshold'] <= 4294967295:
        return self._values['heavy_urls']['latency_threshold']
    raise F5ModuleError("Valid 'latency_threshold' must be in range 0 - 4294967295 milliseconds.")