from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def async_timeout(self):
    divisor = 100
    timeout = self._values['async_timeout']
    if timeout < 150 or timeout > 1800:
        raise F5ModuleError('Timeout value must be between 150 and 1800 seconds.')
    delay = timeout / divisor
    return (delay, divisor)