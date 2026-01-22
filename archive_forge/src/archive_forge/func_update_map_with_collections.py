from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map.resource_map import ResourceMap
def update_map_with_collections(self):
    """Updates the resource map with existing apitools collections."""
    apitools_api_version_map = self.get_apitools_apis()
    apitools_api_names = set(apitools_api_version_map.keys())
    yaml_file_api_names = {api.get_api_name() for api in self._resource_map}
    apis_to_add = apitools_api_names - yaml_file_api_names
    apis_to_update = apitools_api_names & yaml_file_api_names
    apis_to_remove = yaml_file_api_names - apitools_api_names
    for api_name in apis_to_add:
        self.add_api_to_map(api_name, apitools_api_version_map[api_name])
    for api_name in apis_to_update:
        self.update_api_in_map(api_name, apitools_api_version_map[api_name])
    for api_name in apis_to_remove:
        self._resource_map.remove_api(api_name)