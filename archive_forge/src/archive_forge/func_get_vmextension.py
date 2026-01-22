from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_vmextension(self):
    """
        Method calling the Azure SDK to get a VM Extension.
        :return: void
        """
    self.log('Checking if the vm extension {0} is present'.format(self.name))
    found = False
    try:
        response = self.compute_client.virtual_machine_extensions.get(self.resource_group, self.virtual_machine_name, self.name)
        found = True
    except ResourceNotFoundError as e:
        self.log('Did not find vm extension')
    if found:
        return vmextension_to_dict(response)
    else:
        return False