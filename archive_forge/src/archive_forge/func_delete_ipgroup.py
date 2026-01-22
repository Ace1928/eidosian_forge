from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def delete_ipgroup(self):
    try:
        response = self.network_client.ip_groups.begin_delete(resource_group_name=self.resource_group, ip_groups_name=self.name)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error deleting IP group {0} - {1}'.format(self.name, str(exc)))
    return response