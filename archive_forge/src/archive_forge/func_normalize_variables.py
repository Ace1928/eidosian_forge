from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.urls import build_service_uri
from ..module_utils.teem import send_teem
def normalize_variables(self, variables):
    result = []
    for variable in variables:
        tmp = dict(((str(k), str(v)) for k, v in iteritems(variable)))
        if 'encrypted' not in tmp:
            tmp['encrypted'] = 'no'
        if 'value' not in tmp:
            tmp['value'] = ''
        elif tmp['value'] == 'none':
            tmp['value'] = ''
        elif tmp['value'] == 'True':
            tmp['value'] = 'yes'
        elif tmp['value'] == 'False':
            tmp['value'] = 'no'
        elif isinstance(tmp['value'], bool):
            if tmp['value'] is True:
                tmp['value'] = 'yes'
            else:
                tmp['value'] = 'no'
        if tmp['encrypted'] == 'True':
            tmp['encrypted'] = 'yes'
        elif tmp['encrypted'] == 'False':
            tmp['encrypted'] = 'no'
        elif isinstance(tmp['encrypted'], bool):
            if tmp['encrypted'] is True:
                tmp['encrypted'] = 'yes'
            else:
                tmp['encrypted'] = 'no'
        result.append(tmp)
    result = sorted(result, key=lambda k: k['name'])
    return result