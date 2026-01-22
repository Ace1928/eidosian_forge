import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def change_file_name(self, filename):
    """Switch log files."""
    self._filename = filename