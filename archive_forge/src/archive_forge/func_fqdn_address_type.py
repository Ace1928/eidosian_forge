from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def fqdn_address_type(self):
    if self._values['fqdn'] is None:
        return None
    if 'addressFamily' in self._values['fqdn']:
        return str(self._values['fqdn']['addressFamily'])