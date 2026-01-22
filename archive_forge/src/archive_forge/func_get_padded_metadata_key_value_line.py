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
def get_padded_metadata_key_value_line(key_string, value_string, extra_indent=0):
    """Returns metadata line with correct padding."""
    spaces_left_of_value = max(1, LONGEST_METADATA_KEY_LENGTH - len(key_string) + METADATA_LINE_INDENT_LENGTH - extra_indent)
    return '{indent}{key}:{_:>{left_spacing}}{value}'.format(_='', indent=' ' * (METADATA_LINE_INDENT_LENGTH + extra_indent), key=key_string, left_spacing=spaces_left_of_value, value=value_string)