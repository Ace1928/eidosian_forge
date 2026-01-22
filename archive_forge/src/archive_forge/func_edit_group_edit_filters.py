from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def edit_group_edit_filters(self, current_filters, norm_managed_filters, managed_filters_merge_mode, belongsto_filters, belongsto_filters_merge_mode):
    """ Edit a manageiq group filters.

        Returns:
            None if no the group was not updated
            If the group was updated the post body part for updating the group
        """
    filters_updated = False
    new_filters_resource = {}
    current_belongsto_set = current_filters.get('belongsto', set())
    if belongsto_filters:
        new_belongsto_set = set(belongsto_filters)
    else:
        new_belongsto_set = set()
    if current_belongsto_set == new_belongsto_set:
        new_filters_resource['belongsto'] = current_filters['belongsto']
    else:
        if belongsto_filters_merge_mode == 'merge':
            current_belongsto_set.update(new_belongsto_set)
            new_filters_resource['belongsto'] = list(current_belongsto_set)
        else:
            new_filters_resource['belongsto'] = list(new_belongsto_set)
        filters_updated = True
    norm_current_filters = self.manageiq_filters_to_sorted_dict(current_filters)
    if norm_current_filters == norm_managed_filters:
        if 'managed' in current_filters:
            new_filters_resource['managed'] = current_filters['managed']
    else:
        if managed_filters_merge_mode == 'merge':
            merged_dict = self.merge_dict_values(norm_current_filters, norm_managed_filters)
            new_filters_resource['managed'] = self.normalized_managed_tag_filters_to_miq(merged_dict)
        else:
            new_filters_resource['managed'] = self.normalized_managed_tag_filters_to_miq(norm_managed_filters)
        filters_updated = True
    if not filters_updated:
        return None
    return new_filters_resource