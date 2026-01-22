from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def vm_size_is_valid(self):
    """
        Validate self.vm_size against the list of virtual machine sizes available for the account and location.

        :return: boolean
        """
    try:
        sizes = self.compute_client.virtual_machine_sizes.list(self.location)
    except ResourceNotFoundError as exc:
        self.fail('Error retrieving available machine sizes - {0}'.format(str(exc)))
    for size in sizes:
        if size.name == self.vm_size:
            return True
    return False