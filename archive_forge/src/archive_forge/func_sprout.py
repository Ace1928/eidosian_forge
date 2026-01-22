from io import BytesIO
from ... import errors, lockable_files
from ...bzr.bzrdir import BzrDir, BzrDirFormat, BzrDirMetaFormat1
from ...controldir import (ControlDir, Converter, MustHaveWorkingTree,
from ...i18n import gettext
from ...lazy_import import lazy_import
from ...transport import NoSuchFile, get_transport, local
import os
from breezy import (
from breezy.bzr import (
from breezy.plugins.weave_fmt.store.versioned import VersionedFileStore
from breezy.transactions import WriteTransaction
from breezy.plugins.weave_fmt import xml4
def sprout(self, url, revision_id=None, force_new_repo=False, recurse=None, possible_transports=None, accelerator_tree=None, hardlink=False, stacked=False, create_tree_if_local=True, source_branch=None):
    """See ControlDir.sprout()."""
    if source_branch is not None:
        my_branch = self.open_branch()
        if source_branch.base != my_branch.base:
            raise AssertionError('source branch %r is not within %r with branch %r' % (source_branch, self, my_branch))
    if stacked:
        raise _mod_branch.UnstackableBranchFormat(self._format, self.root_transport.base)
    if not create_tree_if_local:
        raise MustHaveWorkingTree(self._format, self.root_transport.base)
    from .workingtree import WorkingTreeFormat2
    self._make_tail(url)
    result = self._format._initialize_for_clone(url)
    try:
        self.open_repository().clone(result, revision_id=revision_id)
    except errors.NoRepositoryPresent:
        pass
    try:
        self.open_branch().sprout(result, revision_id=revision_id)
    except errors.NotBranchError:
        pass
    WorkingTreeFormat2().initialize(result, accelerator_tree=accelerator_tree, hardlink=hardlink)
    return result