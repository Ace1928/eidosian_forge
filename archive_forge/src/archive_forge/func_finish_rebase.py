from ...commands import Command, display_command
from ...errors import (CommandError, ConflictsInTree, NoWorkingTree,
from ...i18n import gettext
from ...option import Option
from ...trace import note
from ...transport import NoSuchFile
def finish_rebase(state, wt, replace_map, replayer):
    from .rebase import rebase
    try:
        rebase(wt.branch.repository, replace_map, replayer)
    except ConflictsInTree:
        raise CommandError(gettext("A conflict occurred replaying a commit. Resolve the conflict and run 'brz rebase-continue' or run 'brz rebase-abort'."))
    state.remove_plan()