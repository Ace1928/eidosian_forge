from __future__ import absolute_import, division, print_function
import copy
import datetime
import traceback
import math
import re
from ansible.module_utils.basic import (
from ansible.module_utils.six import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
@property
def host_board_serial(self):
    if self._values['system-info'] is None:
        return None
    if 'hostBoardSerialNum' not in self._values['system-info'][0]:
        return None
    if self._values['system-info'][0]['hostBoardSerialNum'].strip() == '':
        return None
    return self._values['system-info'][0]['hostBoardSerialNum']