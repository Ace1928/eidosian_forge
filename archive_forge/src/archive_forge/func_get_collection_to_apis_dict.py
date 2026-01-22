from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map.resource_map import ResourceMap
def get_collection_to_apis_dict(self, api_name, api_versions):
    """Gets collection names for all collections in all versions of an api.

    Args:
      api_name: Name of the api to be added.
      api_versions: All registered versions of the api.

    Returns:
      collction_names: Names of every registered apitools collection.
    """
    collection_to_apis_dict = {}
    for version in api_versions:
        resource_collections = [registry.APICollection(c) for c in apis_internal._GetApiCollections(api_name, version)]
        for resource_collection in resource_collections:
            if resource_collection.name in collection_to_apis_dict:
                collection_to_apis_dict[resource_collection.name].append(version)
            else:
                collection_to_apis_dict[resource_collection.name] = [version]
    return collection_to_apis_dict