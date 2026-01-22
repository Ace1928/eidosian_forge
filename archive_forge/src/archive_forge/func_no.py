import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def no(self, controldir, revspec):
    """Mark a given revision as wrong."""
    self._set_state(controldir, revspec, 'no')