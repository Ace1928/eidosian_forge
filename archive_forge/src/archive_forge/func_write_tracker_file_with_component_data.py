from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import hashlib
import json
import os
import re
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import scaled_integer
def write_tracker_file_with_component_data(tracker_file_path, source_object_resource, slice_start_byte=None, total_components=None):
    """Updates or creates a tracker file for component or multipart download.

  Args:
    tracker_file_path (str): The path to the tracker file.
    source_object_resource (resource_reference.ObjectResource): Needed for
      object etag and optionally generation.
    slice_start_byte (int|None): Where to resume downloading from. Signals
      this is the tracker file of a component.
    total_components (int|None): Total number of components in download. Signals
      this is the parent tracker file of a sliced download.
  """
    component_data = {'etag': source_object_resource.etag, 'generation': source_object_resource.generation}
    if slice_start_byte is not None:
        if total_components is not None:
            raise errors.Error('Cannot have a tracker file with slice_start_byte and total_components. slice_start_byte signals a component within a larger operation. total_components signals the parent tracker for a multi-component operation.')
        component_data['slice_start_byte'] = slice_start_byte
    if total_components is not None:
        component_data['total_components'] = total_components
    _write_json_to_tracker_file(tracker_file_path, component_data)