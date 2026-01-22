from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import time
from gslib.thread_message import ProgressMessage
from gslib.utils import parallelism_framework_util
Gathers information describing the operation progress.

    Actual message is printed to stderr by UIThread.

    Args:
      last_byte_processed: The last byte processed in the file. For file
                           components, this number should be in the range
                           [start_byte:start_byte + override_total_size].
      total_size: Total size of the ongoing operation.
    