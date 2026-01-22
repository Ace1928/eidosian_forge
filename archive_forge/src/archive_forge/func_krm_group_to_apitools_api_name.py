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
def krm_group_to_apitools_api_name(krm_group, apitools_api_names):
    if krm_group in apitools_api_names:
        return krm_group
    else:
        for api_name in apitools_api_names:
            if krm_group in api_name:
                if api_name.startswith(krm_group) or api_name.endswith(krm_group):
                    return api_name