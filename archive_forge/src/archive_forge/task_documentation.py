from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import enum
from googlecloudsdk.core.util import debug_output
import six
Task executor calls this method on a completed task before discarding it.

    An example use case is a subclass that needs to report its final status and
    if it failed or succeeded at some operation.

    Args:
      error (Exception|None): Task executor may pass an error object.
      task_status_queue (multiprocessing.Queue): Used by task to report it
        progress to a central location.
    