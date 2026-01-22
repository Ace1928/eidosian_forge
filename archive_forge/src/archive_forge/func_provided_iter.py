import abc
import enum
from taskflow import atom
from taskflow import exceptions as exc
from taskflow.utils import misc
def provided_iter(self):
    """Iterates over all the values the retry has attempted (in order)."""
    for provided, outcomes in self._contents:
        yield provided