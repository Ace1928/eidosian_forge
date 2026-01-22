from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
@staticmethod
def normalize_user_managed_filters_to_sorted_dict(managed_filters, module):
    if not managed_filters:
        return None
    res = {}
    for cat_key in managed_filters:
        cat_array = []
        if not isinstance(managed_filters[cat_key], list):
            module.fail_json(msg='Entry "{0}" of managed_filters must be a list!'.format(cat_key))
        for tags in managed_filters[cat_key]:
            miq_managed_tag = '/managed/' + cat_key + '/' + tags
            cat_array.append(miq_managed_tag)
        if cat_array:
            cat_array.sort()
            res[cat_key] = cat_array
    return res