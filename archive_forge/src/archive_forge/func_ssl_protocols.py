from __future__ import absolute_import, division, print_function
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def ssl_protocols(self):
    default = ' '.join(Parameters._protocols.split(' '))
    if self._values['ssl_protocols'] == default:
        return 'default'
    else:
        return self._values['ssl_protocols']