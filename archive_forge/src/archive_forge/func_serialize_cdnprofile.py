from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import re
def serialize_cdnprofile(self, cdnprofile):
    """
        Convert a CDN profile object to dict.
        :param cdn: CDN profile object
        :return: dict
        """
    result = self.serialize_obj(cdnprofile, AZURE_OBJECT_CLASS)
    new_result = {}
    new_result['id'] = cdnprofile.id
    new_result['resource_group'] = re.sub('\\/.*', '', re.sub('.*resourcegroups\\/', '', result['id']))
    new_result['name'] = cdnprofile.name
    new_result['type'] = cdnprofile.type
    new_result['location'] = cdnprofile.location
    new_result['resource_state'] = cdnprofile.resource_state
    new_result['sku'] = cdnprofile.sku.name
    new_result['provisioning_state'] = cdnprofile.provisioning_state
    new_result['tags'] = cdnprofile.tags
    return new_result