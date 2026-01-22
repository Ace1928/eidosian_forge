from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_smb_client_policies_dict(blade):
    policies_info = {}
    policies = list(blade.get_smb_client_policies().items)
    for policy in range(0, len(policies)):
        policy_name = policies[policy].name
        policies_info[policy_name] = {'local': policies[policy].is_local, 'enabled': policies[policy].enabled, 'version': policies[policy].version, 'rules': []}
        for rule in range(0, len(policies[policy].rules)):
            policies_info[policy_name]['rules'].append({'name': policies[policy].rules[rule].name, 'change': getattr(policies[policy].rules[rule], 'change', None), 'full_control': getattr(policies[policy].rules[rule], 'full_control', None), 'principal': getattr(policies[policy].rules[rule], 'principal', None), 'read': getattr(policies[policy].rules[rule], 'read', None), 'client': getattr(policies[policy].rules[rule], 'client', None), 'index': getattr(policies[policy].rules[rule], 'index', None), 'policy_version': getattr(policies[policy].rules[rule], 'policy_version', None), 'encryption': getattr(policies[policy].rules[rule], 'encryption', None), 'permission': getattr(policies[policy].rules[rule], 'permission', None)})
    return policies_info