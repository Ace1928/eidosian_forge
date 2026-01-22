from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import six
from gslib.exception import CommandException
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.constants import UTF8
Seeks on the buffered stream.

    Args:
      offset: The offset to seek to; must be within the buffer bounds.
      whence: Must be os.SEEK_SET.

    Raises:
      CommandException if an unsupported seek mode or position is used.
    