from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def set_config_flag(self, rest_api, headers):
    body = {'value': True, 'valueType': 'BOOLEAN'}
    base_url = '/occm/api/occm/config/skip-eligibility-paygo-upgrade'
    response, err, dummy = rest_api.put(base_url, body, header=headers)
    if err is not None:
        return (False, 'set_config_flag error')
    return (True, None)