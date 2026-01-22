from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import datetime
import enum
import threading
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import metrics_util
from googlecloudsdk.command_lib.storage import thread_messages
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import scaled_integer
import six
def progress_manager(task_status_queue=None, progress_manager_args=None):
    """Factory function that returns a ProgressManager instance.

  Args:
    task_status_queue (multiprocessing.Queue|None): Tasks can submit their
      progress messages here.
    progress_manager_args (ProgressManagerArgs|None): Determines what type of
      progress indicator to display.

  Returns:
    An instance of _ProgressManager or _NoOpProgressManager.
  """
    if task_status_queue is not None:
        return _ProgressManager(task_status_queue, progress_manager_args)
    else:
        return _NoOpProgressManager()