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
def remove_krm_group(apitools_collection_guess, krm_group):
    """Remove krm_group prefix from krm_kind."""
    if krm_group.lower() in apitools_collection_guess.lower():
        apitools_collection_guess = apitools_collection_guess[len(krm_group):]
    return apitools_collection_guess