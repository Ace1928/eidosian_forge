import abc
import enum
from taskflow import atom
from taskflow import exceptions as exc
from taskflow.utils import misc
def caused_by(self, exception_cls, index=None, include_retry=False):
    """Checks if the exception class provided caused the failures.

        If the index is not provided, then all outcomes are iterated over.

        NOTE(harlowja): only if ``include_retry`` is provided as true (defaults
                        to false) will the potential retries own failure be
                        checked against as well.
        """
    for name, failure in self.outcomes_iter(index=index):
        if failure.check(exception_cls):
            return True
    if include_retry and self._failure is not None:
        if self._failure.check(exception_cls):
            return True
    return False