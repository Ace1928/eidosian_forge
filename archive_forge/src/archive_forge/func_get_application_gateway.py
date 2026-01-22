from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def get_application_gateway(self, id):
    id_dict = parse_resource_id(id)
    try:
        return self.network_client.application_gateways.get(id_dict.get('resource_group', self.resource_group), id_dict.get('name'))
    except ResourceNotFoundError as exc:
        self.fail('Error fetching application_gateway {0} - {1}'.format(id, str(exc)))