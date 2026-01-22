from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def populate_filter_list(self):
    """Populate the filter list"""
    if len(self.module.params.get('gather_subset')) > 1:
        return []
    filters = self.module.params.get('filters') or []
    return [f'{self.filter_mapping.get(filter_dict['filter_operator'])},{filter_dict['filter_key']},{filter_dict['filter_value']}' for filter_dict in filters]