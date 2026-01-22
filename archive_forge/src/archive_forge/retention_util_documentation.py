from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
Converts a retention period string pattern to equivalent seconds.

  Args:
    pattern: a string pattern that represents a retention period.

  Returns:
    Returns the retention period in seconds. If the pattern does not match
  