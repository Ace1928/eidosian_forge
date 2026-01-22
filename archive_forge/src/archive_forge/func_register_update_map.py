from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map.resource_map import ResourceMap
def register_update_map(self, update_map):
    """Registers an update map and map of allowed mismatches while updating.

    Args:
      update_map: Map with an analogous structure to the resource map which
        contains metadata fields and values to apply to the resource map.
    """
    self._update_maps.append(update_map)