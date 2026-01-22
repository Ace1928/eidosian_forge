from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def version_is_greater_or_equal_15(self):
    version = tmos_version(self.client)
    if Version(version) >= Version('15.0.0'):
        return True
    else:
        return False