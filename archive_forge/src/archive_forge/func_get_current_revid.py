import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def get_current_revid(self):
    """Return the current revision id."""
    return self._revid