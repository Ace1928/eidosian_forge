from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def extract_rule_and_section_from_rulebase_response(response):
    section_name = None
    rule = response['rulebase'][0]
    i = 0
    while 'rulebase' in rule and len(rule['rulebase']) == 0:
        i += 1
        rule = response['rulebase'][i]
    while 'rulebase' in rule:
        section_name = rule['name']
        rule = rule['rulebase'][0]
    return (rule, section_name)