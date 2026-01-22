from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core.util import scaled_integer
def replace_autoclass_value_with_prefixed_time(bucket_resource, use_gsutil_time_style=False):
    """Converts raw datetime to 'Enabled on [formatted string]'."""
    datetime_object = getattr(bucket_resource, 'autoclass_enabled_time', None)
    if not datetime_object:
        return
    if use_gsutil_time_style:
        datetime_string = _gsutil_format_datetime_string(datetime_object)
    else:
        datetime_string = resource_util.get_formatted_timestamp_in_utc(datetime_object)
    bucket_resource.autoclass_enabled_time = 'Enabled on ' + datetime_string