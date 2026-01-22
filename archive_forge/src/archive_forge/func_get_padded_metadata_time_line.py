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
def get_padded_metadata_time_line(key_string, value_time):
    """Returns _get_padded_metadata_value_line with formatted time value."""
    formatted_time = get_formatted_timestamp_in_utc(value_time)
    return get_padded_metadata_key_value_line(key_string, formatted_time)