from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def update_cdnendpoint(self):
    """
        Updates a Azure CDN endpoint.

        :return: deserialized Azure CDN endpoint instance state dictionary
        """
    self.log('Updating the Azure CDN endpoint instance {0}'.format(self.name))
    endpoint_update_properties = EndpointUpdateParameters(tags=self.tags, origin_host_header=self.origin_host_header, origin_path=self.origin_path, content_types_to_compress=default_content_types() if self.is_compression_enabled and (not self.content_types_to_compress) else self.content_types_to_compress, is_compression_enabled=self.is_compression_enabled, is_http_allowed=self.is_http_allowed, is_https_allowed=self.is_https_allowed, query_string_caching_behavior=self.query_string_caching_behavior)
    try:
        poller = self.cdn_client.endpoints.begin_update(self.resource_group, self.profile_name, self.name, endpoint_update_properties)
        response = self.get_poller_result(poller)
        return cdnendpoint_to_dict(response)
    except Exception as exc:
        self.log('Error attempting to update Azure CDN endpoint instance.')
        self.fail('Error updating Azure CDN endpoint instance: {0}'.format(exc.message))