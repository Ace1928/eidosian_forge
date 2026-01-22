import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def yes(self, controldir, revspec):
    """Mark that a given revision has the state we're looking for."""
    self._set_state(controldir, revspec, 'yes')