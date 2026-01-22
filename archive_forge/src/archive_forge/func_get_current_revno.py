import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def get_current_revno(self):
    """Return the current revision number as a tuple."""
    return self._branch.revision_id_to_dotted_revno(self._revid)