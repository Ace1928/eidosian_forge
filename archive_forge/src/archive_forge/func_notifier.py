import abc
import os
import threading
import fasteners
from taskflow import engines
from taskflow import exceptions as excp
from taskflow.types import entity
from taskflow.types import notifier
from taskflow.utils import misc
@property
def notifier(self):
    """The conductor actions (or other state changes) notifier.

        NOTE(harlowja): different conductor implementations may emit
        different events + event details at different times, so refer to your
        conductor documentation to know exactly what can and what can not be
        subscribed to.
        """
    return self._notifier