from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def perform_modify(self, sdc_details, sdc_new_name, performance_profile):
    changed = False
    if sdc_new_name is not None and sdc_new_name != sdc_details['name']:
        changed = self.rename_sdc(sdc_details['id'], sdc_new_name)
    if performance_profile and performance_profile != sdc_details['perfProfile']:
        changed = self.set_performance_profile(sdc_details['id'], performance_profile)
    return changed