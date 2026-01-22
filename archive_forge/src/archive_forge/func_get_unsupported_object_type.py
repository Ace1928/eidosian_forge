from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import calendar
import datetime
import enum
import json
import textwrap
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core.resource import resource_projector
def get_unsupported_object_type(resource):
    """Returns unsupported type or None if object is supported for copies.

  Currently, S3 Glacier objects are the only unsupported object type.

  Args:
    resource (ObjectResource|FileObjectResource): Check if this resource is
      supported for copies.

  Returns:
    (UnsupportedObjectType|None) If resource is unsupported, the unsupported
      type, else None.
  """
    if isinstance(resource, resource_reference.ObjectResource) and resource.storage_url.scheme == storage_url.ProviderPrefix.S3 and (resource.storage_class == 'GLACIER'):
        return UnsupportedObjectType.GLACIER
    return None