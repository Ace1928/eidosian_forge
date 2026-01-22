from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def remove_operation_prefix(operation_string):
    """Removes prefix from transfer operation if necessary."""
    if operation_string.startswith(_OPERATIONS_PREFIX_STRING):
        return operation_string[len(_OPERATIONS_PREFIX_STRING):]
    return operation_string