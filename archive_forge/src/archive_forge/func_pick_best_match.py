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
def pick_best_match(possible_matches):
    """Pick best match to our guess for apitools collection."""
    if len(possible_matches) == 1:
        return possible_matches[0]
    elif len(possible_matches) > 1:
        possible_matches = sorted(possible_matches, key=lambda x: len(x.split('.')))
        if len(possible_matches[0].split('.')) < len(possible_matches[1].split('.')):
            return possible_matches[0]
        else:
            for priority_scope in ['locations', 'regions', 'zones']:
                for possible_match in possible_matches:
                    if priority_scope in possible_match:
                        return possible_match
    else:
        return None