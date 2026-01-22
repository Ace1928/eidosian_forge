from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
def create_or_update_hostgroup(self, host_group):
    try:
        response = self.compute_client.dedicated_host_groups.create_or_update(resource_group_name=self.resource_group, host_group_name=self.name, parameters=host_group)
    except Exception as exc:
        self.fail('Error creating or updating host group {0} - {1}'.format(self.name, str(exc)))
    return self.hostgroup_to_dict(response)