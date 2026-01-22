from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def validate_and_get_first_resource_id_uri(module, idrac, base_uri):
    odata = '@odata.id'
    found = False
    res_id_uri = None
    res_id_input = module.params.get('resource_id')
    res_id_members = get_dynamic_uri(idrac, base_uri, 'Members')
    for each in res_id_members:
        if res_id_input and res_id_input == each[odata].split('/')[-1]:
            res_id_uri = each[odata]
            found = True
            break
    if not found and res_id_input:
        return (res_id_uri, INVALID_ID_MSG.format(res_id_input, 'resource_id'))
    elif res_id_input is None:
        res_id_uri = res_id_members[0][odata]
    return (res_id_uri, '')