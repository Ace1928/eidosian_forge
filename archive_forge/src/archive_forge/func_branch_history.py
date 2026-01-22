from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
def branch_history(graph, revid):
    ret = list(graph.iter_lefthand_ancestry(revid, (revision.NULL_REVISION,)))
    ret.reverse()
    return ret