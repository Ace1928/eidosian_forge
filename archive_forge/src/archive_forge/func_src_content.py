from __future__ import absolute_import, division, print_function
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def src_content(self):
    if not os.path.exists(self._values['src']):
        raise F5ModuleError("The specified 'src' was not found.")
    with open(self._values['src']) as f:
        result = f.read()
    return result