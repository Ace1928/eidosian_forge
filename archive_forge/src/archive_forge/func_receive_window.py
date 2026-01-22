from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def receive_window(self):
    window = self._values['receive_window']
    if window is None:
        return None
    if window < 16 or window > 128:
        raise F5ModuleError('Receive Window value must be between 16 and 128')
    return self._values['receive_window']