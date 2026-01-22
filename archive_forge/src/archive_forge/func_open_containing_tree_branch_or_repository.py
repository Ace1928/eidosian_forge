from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
@classmethod
def open_containing_tree_branch_or_repository(klass, location):
    """Return the working tree, branch and repo contained by a location.

        Returns (tree, branch, repository, relpath).
        If there is no tree containing the location, tree will be None.
        If there is no branch containing the location, branch will be None.
        If there is no repository containing the location, repository will be
        None.
        relpath is the portion of the path that is contained by the innermost
        ControlDir.

        If no tree, branch or repository is found, a NotBranchError is raised.
        """
    controldir, relpath = klass.open_containing(location)
    try:
        tree, branch = controldir._get_tree_branch()
    except errors.NotBranchError:
        try:
            repo = controldir.find_repository()
            return (None, None, repo, relpath)
        except errors.NoRepositoryPresent:
            raise errors.NotBranchError(location)
    return (tree, branch, branch.repository, relpath)