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
def auto_detect(self):
    if self._values['heavy_urls'] is None:
        return None
    result = flatten_boolean(self._values['heavy_urls']['auto_detect'])
    if result == 'yes':
        return 'enabled'
    if result == 'no':
        return 'disabled'
    return result