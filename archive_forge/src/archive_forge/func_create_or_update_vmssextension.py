from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_vmssextension(self):
    self.log('Creating VMSS extension {0}'.format(self.name))
    try:
        params = self.compute_models.VirtualMachineScaleSetExtension(location=self.location, publisher=self.publisher, type_properties_type=self.type, type_handler_version=self.type_handler_version, auto_upgrade_minor_version=self.auto_upgrade_minor_version, settings=self.settings, protected_settings=self.protected_settings)
        poller = self.compute_client.virtual_machine_scale_set_extensions.begin_create_or_update(resource_group_name=self.resource_group, vm_scale_set_name=self.vmss_name, vmss_extension_name=self.name, extension_parameters=params)
        response = self.get_poller_result(poller)
        return response.as_dict()
    except Exception as e:
        self.log('Error attempting to create the VMSS extension.')
        self.fail('Error creating the VMSS extension: {0}'.format(str(e)))