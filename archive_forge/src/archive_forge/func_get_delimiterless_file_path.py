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
def get_delimiterless_file_path(file_path):
    """Returns a string representation of the given path without any delimiters.

  Args:
    file_path (str): Path from which delimiters should be removed.

  Returns:
    A copy of file_path without any delimiters.
  """
    return re.sub(RE_DELIMITER_PATTERN, '_', file_path)