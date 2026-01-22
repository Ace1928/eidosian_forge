from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def create_or_update_ipgroup(self, ip_group):
    try:
        response = self.network_client.ip_groups.begin_create_or_update(resource_group_name=self.resource_group, ip_groups_name=self.name, parameters=ip_group)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error creating or updating IP group {0} - {1}'.format(self.name, str(exc)))
    return self.ipgroup_to_dict(response)