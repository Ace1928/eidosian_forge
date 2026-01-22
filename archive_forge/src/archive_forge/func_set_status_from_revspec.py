import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def set_status_from_revspec(self, revspec, status):
    """Set the bisection status for the revision in revspec."""
    self._load_tree()
    revid = revspec[0].in_history(self._branch).rev_id
    self._set_status(revid, status)