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
def require_stacking(self, stack_on=None, possible_transports=None, _skip_repo=False):
    """We have a request to stack, try to ensure the formats support it.

        :param stack_on: If supplied, it is the URL to a branch that we want to
            stack on. Check to see if that format supports stacking before
            forcing an upgrade.
        """
    new_repo_format = None
    new_branch_format = None
    target = [None, False, None]

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
    if not _skip_repo and (not self.repository_format.supports_external_lookups):
        target_branch, _, do_upgrade = get_target_branch()
        if target_branch is None:
            if do_upgrade:
                if self.repository_format.rich_root_data:
                    new_repo_format = knitpack_repo.RepositoryFormatKnitPack5RichRoot()
                else:
                    new_repo_format = knitpack_repo.RepositoryFormatKnitPack5()
        else:
            new_repo_format = target_branch.repository._format
            if not new_repo_format.supports_external_lookups:
                new_repo_format = None
        if new_repo_format is not None:
            self.repository_format = new_repo_format
            note(gettext('Source repository format does not support stacking, using format:\n  %s'), new_repo_format.get_format_description())
    if not self.get_branch_format().supports_stacking():
        target_branch, _, do_upgrade = get_target_branch()
        if target_branch is None:
            if do_upgrade:
                from .branch import BzrBranchFormat7
                new_branch_format = BzrBranchFormat7()
        else:
            new_branch_format = target_branch._format
            if not new_branch_format.supports_stacking():
                new_branch_format = None
        if new_branch_format is not None:
            self.set_branch_format(new_branch_format)
            note(gettext('Source branch format does not support stacking, using format:\n  %s'), new_branch_format.get_format_description())