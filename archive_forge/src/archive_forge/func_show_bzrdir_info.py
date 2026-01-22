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
def show_bzrdir_info(a_controldir, verbose=False, outfile=None):
    """Output to stdout the 'info' for a_controldir."""
    if outfile is None:
        outfile = sys.stdout
    try:
        tree = a_controldir.open_workingtree(recommend_upgrade=False)
    except (NoWorkingTree, NotLocalUrl, NotBranchError):
        tree = None
        try:
            branch = a_controldir.open_branch(name='')
        except NotBranchError:
            branch = None
            try:
                repository = a_controldir.open_repository()
            except NoRepositoryPresent:
                lockable = None
                repository = None
            else:
                lockable = repository
        else:
            repository = branch.repository
            lockable = branch
    else:
        branch = tree.branch
        repository = branch.repository
        lockable = tree
    if lockable is not None:
        lockable.lock_read()
    try:
        show_component_info(a_controldir, repository, branch, tree, verbose, outfile)
    finally:
        if lockable is not None:
            lockable.unlock()