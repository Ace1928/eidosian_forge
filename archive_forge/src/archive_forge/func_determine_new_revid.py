from ...commands import Command, display_command
from ...errors import (CommandError, ConflictsInTree, NoWorkingTree,
from ...i18n import gettext
from ...option import Option
from ...trace import note
from ...transport import NoSuchFile
def determine_new_revid(old_revid, new_parents):
    return create_deterministic_revid(old_revid, new_parents)