from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
@staticmethod
def instrument_branch(branch, gets):
    old_get = branch._transport.get

    def get(*args, **kwargs):
        gets.append((args, kwargs))
        return old_get(*args, **kwargs)
    branch._transport.get = get