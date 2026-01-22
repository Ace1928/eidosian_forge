from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import re
def serialize_cdnendpoint(self, cdnendpoint):
    """
        Convert a Azure CDN endpoint object to dict.
        :param cdn: Azure CDN endpoint object
        :return: dict
        """
    result = self.serialize_obj(cdnendpoint, AZURE_OBJECT_CLASS)
    new_result = {}
    new_result['id'] = cdnendpoint.id
    new_result['resource_group'] = re.sub('\\/.*', '', re.sub('.*resourcegroups\\/', '', result['id']))
    new_result['profile_name'] = re.sub('\\/.*', '', re.sub('.*profiles\\/', '', result['id']))
    new_result['name'] = cdnendpoint.name
    new_result['type'] = cdnendpoint.type
    new_result['location'] = cdnendpoint.location
    new_result['resource_state'] = cdnendpoint.resource_state
    new_result['provisioning_state'] = cdnendpoint.provisioning_state
    new_result['query_string_caching_behavior'] = cdnendpoint.query_string_caching_behavior
    new_result['is_compression_enabled'] = cdnendpoint.is_compression_enabled
    new_result['is_http_allowed'] = cdnendpoint.is_http_allowed
    new_result['is_https_allowed'] = cdnendpoint.is_https_allowed
    new_result['content_types_to_compress'] = cdnendpoint.content_types_to_compress
    new_result['origin_host_header'] = cdnendpoint.origin_host_header
    new_result['origin_path'] = cdnendpoint.origin_path
    new_result['origin'] = dict(name=cdnendpoint.origins[0].name, host_name=cdnendpoint.origins[0].host_name, http_port=cdnendpoint.origins[0].http_port, https_port=cdnendpoint.origins[0].https_port)
    new_result['tags'] = cdnendpoint.tags
    return new_result