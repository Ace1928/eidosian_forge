import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
def revert_task(self, task, task_uuid, arguments, result, failures, progress_callback=None):
    return self._submit_task(_revert_task, task, arguments, result, failures, progress_callback=progress_callback)