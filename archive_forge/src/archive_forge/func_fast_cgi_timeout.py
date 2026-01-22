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
def fast_cgi_timeout(self):
    if self._values['fast_cgi_timeout'] is None:
        return None
    return int(self._values['fast_cgi_timeout'])