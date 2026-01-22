from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def gen_scale_in_policy(self):
    if self.scale_in_policy is None:
        return None
    return self.compute_models.ScaleInPolicy(rules=[self.scale_in_policy])