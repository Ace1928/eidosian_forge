from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.storage import errors
def get_operation_bucket_and_id_from_name(operation_name):
    """Extracts operation ID from user input of operation name or ID."""
    m = re.match(_BUCKET_AND_ID_OPERATION_NAME_REGEX, operation_name)
    try:
        return (m.group('bucket'), m.group('id'))
    except AttributeError:
        raise errors.Error('Invalid operation name format. Expected: {} Received: {}'.format(_BUCKET_AND_ID_OPERATION_NAME_REGEX, operation_name))