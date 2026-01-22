from __future__ import absolute_import, division, print_function
import re
import time
import xml.etree.ElementTree
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import iControlRestSession
from ..module_utils.teem import send_teem
def wait_for_mcpd(self):
    nops = 0
    time.sleep(5)
    while nops < 4:
        try:
            if self._is_mcpd_ready_on_device():
                nops += 1
            else:
                nops = 0
        except Exception as ex:
            if '"message":"X-F5-Auth-Token has expired."' in str(ex):
                raise F5ModuleError('X-F5-Auth-Token has expired.')
            pass
        time.sleep(5)