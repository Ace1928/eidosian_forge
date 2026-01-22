import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
def revert_retry(self, retry, arguments):
    """Schedules retry reversion."""
    fut = self._executor.submit(_revert_retry, retry, arguments)
    fut.atom = retry
    return fut