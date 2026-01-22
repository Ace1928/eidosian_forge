from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def build_rulebase_command(api_call_object):
    rulebase_command = 'show-' + api_call_object.split('-')[0] + '-rulebase'
    if api_call_object == 'threat-exception':
        rulebase_command = 'show-threat-rule-exception-rulebase'
    return rulebase_command