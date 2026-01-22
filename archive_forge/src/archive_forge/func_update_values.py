from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
import copy
def update_values(self, existing_values, param_values, append):
    new_values = copy.copy(existing_values)
    changed = False
    for item in param_values:
        if item not in new_values:
            changed = True
            new_values.append(item)
    if not append:
        for item in existing_values:
            if item not in param_values:
                new_values.remove(item)
                changed = True
    return (changed, new_values)