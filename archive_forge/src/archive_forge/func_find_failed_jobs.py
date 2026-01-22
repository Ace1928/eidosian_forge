from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def find_failed_jobs(resp, rest_obj):
    msg, fail = ('Export log job completed with errors.', False)
    history = rest_obj.invoke_request('GET', EXE_HISTORY_URI.format(resp['Id']))
    if history.json_data['value']:
        hist = history.json_data['value'][0]
        history_details = rest_obj.invoke_request('GET', '{0}({1})/ExecutionHistoryDetails'.format(EXE_HISTORY_URI.format(resp['Id']), hist['Id']))
        for hd in history_details.json_data['value']:
            if not re.findall('Job status for JID_\\d+ is Completed with Errors.', hd['Value']):
                fail = True
                break
        else:
            fail = False
    return (msg, fail)