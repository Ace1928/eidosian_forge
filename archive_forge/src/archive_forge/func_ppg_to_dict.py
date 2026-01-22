from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def ppg_to_dict(self, ppg):
    result = dict(id=ppg.id, name=ppg.name, location=ppg.location, tags=ppg.tags, proximity_placement_group_type=ppg.proximity_placement_group_type, virtual_machines=[dict(id=x.id) for x in ppg.virtual_machines], virtual_machine_scale_sets=[dict(id=x.id) for x in ppg.virtual_machine_scale_sets], availability_sets=[dict(id=x.id) for x in ppg.availability_sets])
    return result