from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_availabilityset(self):
    """
        Method calling the Azure SDK to get an AS.
        :return: void
        """
    self.log('Checking if the availabilityset {0} is present'.format(self.name))
    found = False
    try:
        response = self.compute_client.availability_sets.get(self.resource_group, self.name)
        found = True
    except ResourceNotFoundError as e:
        self.log('Did not find the Availability set.')
    if found is True:
        return availability_set_to_dict(response)
    else:
        return False