from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import math
import os
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import scaled_integer
def get_component_count(file_size, target_component_size, max_components):
    """Returns the # components a file would be split into for a composite upload.

  Args:
    file_size (int|None): Total byte size of file being divided into components.
      None if could not be determined.
    target_component_size (int|str): Target size for each component if not total
      components isn't capped by max_components. May be byte count int or size
      string (e.g. "50M").
    max_components (int|None): Limit on allowed components regardless of
      file_size and target_component_size. None indicates no limit.

  Returns:
    int: Number of components to split file into for composite upload.
  """
    if file_size is None:
        return 1
    if isinstance(target_component_size, int):
        target_component_size_bytes = target_component_size
    else:
        target_component_size_bytes = scaled_integer.ParseInteger(target_component_size)
    return min(math.ceil(file_size / target_component_size_bytes), max_components if max_components is not None else float('inf'))