from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_correlator_rule(config_data):
    rule_name = config_data.get('rule_name')
    command = 'snmp-server correlator rule {rule_name}'.format(rule_name=rule_name)
    if config_data.get('timeout'):
        command += ' timeout {timeout}'.format(timeout=config_data['timeout'])
    return command