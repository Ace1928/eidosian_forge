from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.declarative.clients import kcc_client
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map import resource_map_update_util
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import name_parsing
def generate_cc_update_map():
    """Generates the map used to update the resource map with cc metadata.

  The returned update map will have an analogous structure to the resource map.
  Each resource will contain the associated metadata values to be applied to the
  resource map.

  Raises:
    KrmToApitoolsResourceNameError: Raised if mismatches occur that are not
      present in _ALLOWED_MISMATCHES.

  Returns:
    Update map containing the config connector support metadata.
  """
    config_connector_data = kcc_client.KccClient().ListResources()
    apitools_resource_map = build_collection_map()
    update_map = {}
    resources_already_seen = set()
    unmatched_resources = set()
    for resource_spec in config_connector_data:
        krm_group = resource_spec['GVK']['Group'].split('.')[0]
        krm_kind = resource_spec['GVK']['Kind']
        apitools_api_name = krm_group_to_apitools_api_name(krm_group, apitools_resource_map.keys())
        try:
            apitools_collection_name = krm_kind_to_apitools_collection_name(krm_kind, krm_group, set(apitools_resource_map[apitools_api_name]))
        except KrmToApitoolsResourceNameError:
            if (krm_group, krm_kind) not in _ALLOWED_MISMATCHES:
                unmatched_resources.add((krm_group, krm_kind))
            continue
        if (apitools_api_name, apitools_collection_name) in resources_already_seen:
            if not resource_spec['ResourceNameFormat']:
                continue
        resources_already_seen.add((apitools_api_name, apitools_collection_name))
        asset_inventory_api_name = apitools_api_name
        asset_inventory_resource_name = krm_kind
        if krm_group in asset_inventory_resource_name.lower():
            asset_inventory_resource_name = asset_inventory_resource_name[len(krm_group):]
        asset_inventory_type = '{}.googleapis.com/{}'.format(asset_inventory_api_name, asset_inventory_resource_name)
        bulk_support = resource_spec['SupportsBulkExport']
        single_export_support = resource_spec['SupportsExport']
        iam_support = resource_spec['SupportsIAM']
        if apitools_api_name not in update_map:
            update_map[apitools_api_name] = {}
        if apitools_collection_name not in update_map[apitools_api_name]:
            update_map[apitools_api_name][apitools_collection_name] = {'support_bulk_export': False, 'support_single_export': False, 'support_iam': False}
        update_map[apitools_api_name][apitools_collection_name]['krm_kind'] = krm_kind
        update_map[apitools_api_name][apitools_collection_name]['krm_group'] = krm_group
        update_map[apitools_api_name][apitools_collection_name]['asset_inventory_type'] = asset_inventory_type
        update_map[apitools_api_name][apitools_collection_name]['support_bulk_export'] = bool(bulk_support)
        update_map[apitools_api_name][apitools_collection_name]['support_single_export'] = bool(single_export_support)
        update_map[apitools_api_name][apitools_collection_name]['support_iam'] = bool(iam_support)
    if unmatched_resources:
        raise KrmToApitoolsResourceNameError('The KRM resources were unable to be matched to apitools collections: {}'.format(unmatched_resources))
    return update_map