from __future__ import absolute_import, division, print_function
import os
import re
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def wait_for_rest_api_restart(self):
    time.sleep(5)
    for x in range(0, 60):
        try:
            self.client.reconnect()
            break
        except Exception:
            time.sleep(3)