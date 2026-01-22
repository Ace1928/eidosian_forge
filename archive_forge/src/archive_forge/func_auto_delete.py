from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def auto_delete(self):
    result = flatten_boolean(self._values['auto_delete'])
    if result == 'yes':
        return 'true'
    if result == 'no':
        return 'false'