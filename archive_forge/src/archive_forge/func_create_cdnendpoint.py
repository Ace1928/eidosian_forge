from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_cdnendpoint(self):
    """
        Creates a Azure CDN endpoint.

        :return: deserialized Azure CDN endpoint instance state dictionary
        """
    self.log('Creating the Azure CDN endpoint instance {0}'.format(self.name))
    origins = []
    for item in self.origins:
        origins.append(DeepCreatedOrigin(name=item['name'], host_name=item['host_name'], http_port=item['http_port'] if 'http_port' in item else None, https_port=item['https_port'] if 'https_port' in item else None))
    parameters = Endpoint(origins=origins, location=self.location, tags=self.tags, origin_host_header=self.origin_host_header, origin_path=self.origin_path, content_types_to_compress=default_content_types() if self.is_compression_enabled and (not self.content_types_to_compress) else self.content_types_to_compress, is_compression_enabled=self.is_compression_enabled if self.is_compression_enabled is not None else False, is_http_allowed=self.is_http_allowed if self.is_http_allowed is not None else True, is_https_allowed=self.is_https_allowed if self.is_https_allowed is not None else True, query_string_caching_behavior=self.query_string_caching_behavior if self.query_string_caching_behavior else QueryStringCachingBehavior.ignore_query_string)
    try:
        poller = self.cdn_client.endpoints.begin_create(self.resource_group, self.profile_name, self.name, parameters)
        response = self.get_poller_result(poller)
        return cdnendpoint_to_dict(response)
    except Exception as exc:
        self.log('Error attempting to create Azure CDN endpoint instance.')
        self.fail('Error creating Azure CDN endpoint instance: {0}'.format(exc.message))