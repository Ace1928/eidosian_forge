from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def create_or_update_pip(self, pip):
    try:
        poller = self.network_client.public_ip_addresses.begin_create_or_update(self.resource_group, self.name, pip)
        pip = self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error creating or updating {0} - {1}'.format(self.name, str(exc)))
    return pip_to_dict(pip)