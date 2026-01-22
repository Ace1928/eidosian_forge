from __future__ import absolute_import, division, print_function
import os
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def version_is_less_than_14(self, version):
    if Version(version) < Version('14.0.0'):
        return True
    else:
        return False