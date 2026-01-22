from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_nfs_export_policies_dict(blade):
    policies_info = {}
    policies = list(blade.get_nfs_export_policies().items)
    for policy in range(0, len(policies)):
        policy_name = policies[policy].name
        policies_info[policy_name] = {'local': policies[policy].is_local, 'enabled': policies[policy].enabled, 'rules': []}
        for rule in range(0, len(policies[policy].rules)):
            policies_info[policy_name]['rules'].append({'access': policies[policy].rules[rule].access, 'anongid': policies[policy].rules[rule].anongid, 'anonuid': policies[policy].rules[rule].anonuid, 'atime': policies[policy].rules[rule].atime, 'client': policies[policy].rules[rule].client, 'fileid_32bit': policies[policy].rules[rule].fileid_32bit, 'permission': policies[policy].rules[rule].permission, 'secure': policies[policy].rules[rule].secure, 'security': policies[policy].rules[rule].security, 'index': policies[policy].rules[rule].index})
    return policies_info