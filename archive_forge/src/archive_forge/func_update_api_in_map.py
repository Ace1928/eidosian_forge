from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map.resource_map import ResourceMap
def update_api_in_map(self, api_name, api_versions):
    """Updates resources in an existing API in the ResourceMap.

    Args:
      api_name: Name of the api to be added.
      api_versions: All registered versions of the api.
    """
    api_data = self._resource_map.get_api(api_name)
    collection_to_apis_dict = self.get_collection_to_apis_dict(api_name, api_versions)
    collection_names = set(collection_to_apis_dict.keys())
    map_resource_names = {resource.get_resource_name() for resource in api_data}
    resources_to_add = collection_names - map_resource_names
    resources_to_update = collection_names & map_resource_names
    resources_to_remove = map_resource_names - collection_names
    for resource_name in resources_to_add:
        supported_apis = collection_to_apis_dict[resource_name]
        api_data.add_resource(base.ResourceData(resource_name, api_name, {'supported_apis': supported_apis}))
    for resource_name in resources_to_update:
        supported_apis = collection_to_apis_dict[resource_name]
        resource_data = api_data.get_resource(resource_name)
        if 'supported_apis' in resource_data:
            resource_data.update_metadata('supported_apis', supported_apis)
        else:
            resource_data.add_metadata('supported_apis', supported_apis)
    for resource_name in resources_to_remove:
        api_data.remove_resource(resource_name)