import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
def set_task_progress(self, task_name, progress, details=None):
    """Set a tasks progress.

        :param task_name: task name
        :param progress: tasks progress (0.0 <-> 1.0)
        :param details: any task specific progress details
        """
    update_with = {META_PROGRESS: progress}
    if details is not None:
        if details:
            update_with[META_PROGRESS_DETAILS] = {'at_progress': progress, 'details': details}
        else:
            update_with[META_PROGRESS_DETAILS] = None
    self._update_atom_metadata(task_name, update_with, expected_type=models.TaskDetail)