from __future__ import absolute_import, division, print_function
import os
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def retain_inheritance_settings(self):
    result = flatten_boolean(self._values['retain_inheritance_settings'])
    return result