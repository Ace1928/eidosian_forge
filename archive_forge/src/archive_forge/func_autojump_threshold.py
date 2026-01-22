import time
from math import inf
from .. import _core
from .._abc import Clock
from .._util import final
from ._run import GLOBAL_RUN_CONTEXT
@autojump_threshold.setter
def autojump_threshold(self, new_autojump_threshold: float) -> None:
    self._autojump_threshold = float(new_autojump_threshold)
    self._try_resync_autojump_threshold()