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
def get_display_dict_for_resource(resource, display_titles_and_defaults, display_raw_keys):
    """Makes a resource better for returning from describe and list commands.

  Display = Removes complex nested objects and makes other string tweaks.

  Args:
    resource (resource_reference.Resource): Resource to format.
    display_titles_and_defaults (namedtuple): Contains names of fields for
      display.
    display_raw_keys (bool): Displays raw API responses if True, otherwise
      standardizes metadata keys. If True, `resource` must have a metadata
      attribute.

  Returns:
    Dictionary representing input resource with optimizations described above.
  """
    if display_raw_keys:
        display_data = resource.metadata
    else:
        display_data = {'storage_url': resource.storage_url.url_string}
        formatted_acl_dict = resource.get_formatted_acl()
        for field in display_titles_and_defaults._fields:
            if field in formatted_acl_dict:
                value = formatted_acl_dict.get(field)
            else:
                value = getattr(resource, field, None)
            display_data[field] = convert_to_json_parsable_type(value)
    return resource_projector.MakeSerializable(display_data)