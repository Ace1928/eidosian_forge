from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def get_relevant_show_rulebase_identifier_payload(api_call_object, payload):
    if api_call_object == 'nat-rule':
        show_rulebase_payload = {'package': payload['package']}
    else:
        show_rulebase_payload = {'name': payload['layer']}
    if api_call_object == 'threat-exception':
        show_rulebase_payload['rule-name'] = payload['rule-name']
    return show_rulebase_payload