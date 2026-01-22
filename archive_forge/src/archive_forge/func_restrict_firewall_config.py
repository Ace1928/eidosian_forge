from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native, to_text
def restrict_firewall_config(config):
    result = restrict_dict(config, ['port', 'status', 'filter_ipv6', 'whitelist_hos'])
    result['rules'] = dict()
    for ruleset in RULES:
        result['rules'][ruleset] = [restrict_dict(rule, RULE_OPTION_NAMES) for rule in config['rules'].get(ruleset) or []]
    return result