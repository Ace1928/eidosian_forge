from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
@staticmethod
def manageiq_filters_to_sorted_dict(current_filters):
    current_managed_filters = current_filters.get('managed')
    if not current_managed_filters:
        return None
    res = {}
    for tag_list in current_managed_filters:
        tag_list.sort()
        key = tag_list[0].split('/')[2]
        res[key] = tag_list
    return res