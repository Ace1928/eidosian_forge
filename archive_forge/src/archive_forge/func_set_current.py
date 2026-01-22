import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def set_current(self, status):
    """Set the current revision to the given bisection status."""
    self._set_status(self._current.get_current_revid(), status)