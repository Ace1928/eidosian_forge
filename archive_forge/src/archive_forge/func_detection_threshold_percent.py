from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def detection_threshold_percent(self):
    if self._values['detection_threshold_percent'] in [None, 'infinite']:
        return self._values['detection_threshold_percent']
    return int(self._values['detection_threshold_percent'])