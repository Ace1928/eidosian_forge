from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def prepare_rule_params_for_execute_module(rule, module_args, position, below_rule_name):
    rule['layer'] = module_args['layer']
    if 'details_level' in module_args.keys():
        rule['details_level'] = module_args['details_level']
    if 'state' not in rule.keys() or ('state' in rule.keys() and rule['state'] != 'absent'):
        if below_rule_name:
            relative_position = {'relative_position': {'below': below_rule_name}}
            rule.update(relative_position)
        else:
            rule['position'] = position
        position = position + 1
        below_rule_name = rule['name']
    return (rule, position, below_rule_name)