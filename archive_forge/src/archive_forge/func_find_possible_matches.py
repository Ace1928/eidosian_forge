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
def find_possible_matches(apitools_collection_guess, apitools_collection_names):
    """Find any apitools collections that reasonably match our guess."""
    possible_matches = []
    for apitools_collection_name in apitools_collection_names:
        split_collection_name = apitools_collection_name.split('.')
        if apitools_collection_guess.lower() in split_collection_name[-1].lower() or split_collection_name[-1].lower() in apitools_collection_guess.lower():
            possible_matches.append(apitools_collection_name)
    return possible_matches