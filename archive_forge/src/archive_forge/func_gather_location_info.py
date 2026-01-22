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
def gather_location_info(repository=None, branch=None, working=None, control=None):
    locs = {}
    if branch is not None:
        branch_path = branch.user_url
        master_path = branch.get_bound_location()
        if master_path is None:
            master_path = branch_path
    else:
        branch_path = None
        master_path = None
        try:
            if control is not None and control.get_branch_reference():
                locs['checkout of branch'] = control.get_branch_reference()
        except NotBranchError:
            pass
    if working:
        working_path = working.user_url
        if working_path != branch_path:
            locs['light checkout root'] = working_path
        if master_path != branch_path:
            if repository.is_shared():
                locs['repository checkout root'] = branch_path
            else:
                locs['checkout root'] = branch_path
        if working_path != master_path:
            master_path_base, params = urlutils.split_segment_parameters(master_path)
            if working_path == master_path_base:
                locs['checkout of co-located branch'] = params['branch']
            elif 'branch' in params:
                locs['checkout of branch'] = '{}, branch {}'.format(master_path_base, params['branch'])
            else:
                locs['checkout of branch'] = master_path
        elif repository.is_shared():
            locs['repository branch'] = branch_path
        elif branch_path is not None:
            locs['branch root'] = branch_path
    else:
        working_path = None
        if repository is not None and repository.is_shared():
            if branch_path is not None:
                locs['repository branch'] = branch_path
        elif branch_path is not None:
            locs['branch root'] = branch_path
        elif repository is not None:
            locs['repository'] = repository.user_url
        elif control is not None:
            locs['control directory'] = control.user_url
        else:
            pass
        if master_path != branch_path:
            locs['bound to branch'] = master_path
    if repository is not None and repository.is_shared():
        locs['shared repository'] = repository.user_url
    order = ['control directory', 'light checkout root', 'repository checkout root', 'checkout root', 'checkout of branch', 'checkout of co-located branch', 'shared repository', 'repository', 'repository branch', 'branch root', 'bound to branch']
    return [(n, locs[n]) for n in order if n in locs]