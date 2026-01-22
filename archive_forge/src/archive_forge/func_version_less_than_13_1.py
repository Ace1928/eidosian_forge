from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def version_less_than_13_1(self):
    version = tmos_version(self.client)
    if Version(version) < Version('13.1.0'):
        return True
    return False