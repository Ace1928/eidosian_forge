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
def get_fault(self):
    result = dict()
    self.set_result_for_license_fault(result)
    self.set_result_for_general_fault(result)
    if 'faultNumber' not in result:
        result['faultNumber'] = None
    return result