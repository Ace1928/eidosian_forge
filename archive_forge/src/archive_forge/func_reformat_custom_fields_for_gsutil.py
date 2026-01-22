from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core.util import scaled_integer
def reformat_custom_fields_for_gsutil(object_resource):
    """Reformats custom metadata full format string in gsutil style."""
    metadata = object_resource.custom_fields
    if not metadata:
        return
    if isinstance(metadata, dict):
        iterable_metadata = metadata.items()
    else:
        iterable_metadata = [(d['key'], d['value']) for d in metadata]
    metadata_lines = []
    for k, v in iterable_metadata:
        metadata_lines.append(resource_util.get_padded_metadata_key_value_line(k, v, extra_indent=2))
    object_resource.custom_fields = '\n' + '\n'.join(metadata_lines)