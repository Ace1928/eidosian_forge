import sys
import time
from io import StringIO
from . import branch as _mod_branch
from . import controldir, errors
from . import hooks as _mod_hooks
from . import osutils, urlutils
from .bzr import bzrdir
from .errors import (NoRepositoryPresent, NotBranchError, NotLocalUrl,
from .missing import find_unmerged
def describe_format(control, repository, branch, tree):
    """Determine the format of an existing control directory

    Several candidates may be found.  If so, the names are returned as a
    single string, separated by ' or '.

    If no matching candidate is found, "unnamed" is returned.
    """
    candidates = []
    if branch is not None and tree is not None and (branch.user_url != tree.user_url):
        branch = None
        repository = None
    non_aliases = set(controldir.format_registry.keys())
    non_aliases.difference_update(controldir.format_registry.aliases())
    for key in non_aliases:
        format = controldir.format_registry.make_controldir(key)
        if isinstance(format, bzrdir.BzrDirMetaFormat1):
            if tree and format.workingtree_format != tree._format:
                continue
            if branch and format.get_branch_format() != branch._format:
                continue
            if repository and format.repository_format != repository._format:
                continue
        if format.__class__ is not control._format.__class__:
            continue
        candidates.append(key)
    if len(candidates) == 0:
        return 'unnamed'
    candidates.sort()
    new_candidates = [c for c in candidates if not controldir.format_registry.get_info(c).hidden]
    if len(new_candidates) > 0:
        candidates = new_candidates
    return ' or '.join(candidates)