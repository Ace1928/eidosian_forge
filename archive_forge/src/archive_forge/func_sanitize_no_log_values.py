from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def sanitize_no_log_values(meraki):
    try:
        meraki.result['diff']['before']['shared_secret'] = 'VALUE_SPECIFIED_IN_NO_LOG_PARAMETER'
    except KeyError:
        pass
    try:
        for i in meraki.result['data']:
            i['shared_secret'] = 'VALUE_SPECIFIED_IN_NO_LOG_PARAMETER'
    except (KeyError, TypeError):
        pass
    try:
        meraki.result['data']['shared_secret'] = 'VALUE_SPECIFIED_IN_NO_LOG_PARAMETER'
    except (KeyError, TypeError):
        pass