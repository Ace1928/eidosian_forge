from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def serialize_vmss(self, vmss):
    """
        Convert a VirtualMachineScaleSet object to dict.

        :param vm: VirtualMachineScaleSet object
        :return: dict
        """
    result = self.serialize_obj(vmss, AZURE_OBJECT_CLASS, enum_modules=AZURE_ENUM_MODULES)
    result['id'] = vmss.id
    result['name'] = vmss.name
    result['type'] = vmss.type
    result['location'] = vmss.location
    result['tags'] = vmss.tags
    return result