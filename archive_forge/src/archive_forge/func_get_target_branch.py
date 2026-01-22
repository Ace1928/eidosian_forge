import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
def get_target_branch():
    if target[1]:
        return target
    if stack_on is None:
        target[:] = [None, True, True]
        return target
    try:
        target_dir = BzrDir.open(stack_on, possible_transports=possible_transports)
    except errors.NotBranchError:
        target[:] = [None, True, False]
        return target
    except errors.JailBreak:
        target[:] = [None, True, True]
        return target
    try:
        target_branch = target_dir.open_branch()
    except errors.NotBranchError:
        target[:] = [None, True, False]
        return target
    target[:] = [target_branch, True, False]
    return target