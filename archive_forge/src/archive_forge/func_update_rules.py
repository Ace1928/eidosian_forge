from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native, to_text
def update_rules(before, after, params, ruleset):
    before_rules = before['rules'][ruleset]
    after_rules = after['rules'][ruleset]
    params_rules = params['rules'][ruleset]
    changed = len(before_rules) != len(params_rules)
    for no, rule in enumerate(params_rules):
        rule['src_ip'] = normalize_ip(rule['src_ip'], rule['ip_version'])
        rule['dst_ip'] = normalize_ip(rule['dst_ip'], rule['ip_version'])
        if no < len(before_rules):
            before_rule = before_rules[no]
            before_rule['src_ip'] = normalize_ip(before_rule['src_ip'], before_rule['ip_version'])
            before_rule['dst_ip'] = normalize_ip(before_rule['dst_ip'], before_rule['ip_version'])
            if before_rule != rule:
                changed = True
        after_rules.append(rule)
    return changed