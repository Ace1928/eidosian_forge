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
def get_exists_string(item):
    """Returns string showing if item exists. May return 'None', '[]', etc."""
    if item or should_preserve_falsy_metadata_value(item):
        return 'Present'
    else:
        return str(item)