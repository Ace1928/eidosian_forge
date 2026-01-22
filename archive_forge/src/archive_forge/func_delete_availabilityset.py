from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_availabilityset(self):
    """
        Method calling the Azure SDK to delete the AS.
        :return: void
        """
    self.log('Deleting availabilityset {0}'.format(self.name))
    try:
        response = self.compute_client.availability_sets.delete(self.resource_group, self.name)
    except Exception as e:
        self.log('Error attempting to delete the availability set.')
        self.fail('Error deleting the availability set: {0}'.format(str(e)))
    return True