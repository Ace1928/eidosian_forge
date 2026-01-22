from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_vmextension(self):
    """
        Method calling the Azure SDK to delete the VM Extension.
        :return: void
        """
    self.log('Deleting vmextension {0}'.format(self.name))
    try:
        poller = self.compute_client.virtual_machine_extensions.begin_delete(self.resource_group, self.virtual_machine_name, self.name)
        self.get_poller_result(poller)
    except Exception as e:
        self.log('Error attempting to delete the vmextension.')
        self.fail('Error deleting the vmextension: {0}'.format(str(e)))