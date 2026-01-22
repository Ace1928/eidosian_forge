from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def device_is_ready(self):
    try:
        if self._is_rest_available():
            return True
    except Exception as ex:
        if 'Failed to validate the SSL' in str(ex):
            raise F5ModuleError(str(ex))
        pass
    return False