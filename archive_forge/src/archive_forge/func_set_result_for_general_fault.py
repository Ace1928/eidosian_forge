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
def set_result_for_general_fault(self, result):
    namespaces = {'ns2': 'http://schemas.xmlsoap.org/soap/envelope/'}
    root = self.content.findall('.//ns2:Fault', namespaces)
    if len(root) == 0:
        return None
    for elem in root[0]:
        if elem.tag == 'faultstring':
            result['faultText'] = elem.text