from breezy import branch, delta, errors, revision, transport
from breezy.tests import per_branch
def hook_func(local, master, old_revno, old_revid, new_revno, new_revid, tree_delta, future_tree):
    raise PreCommitException(new_revid)