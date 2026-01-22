import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def make_diverged_tree_and_upload_location(self):
    tree_a = self.make_branch_and_tree('tree_a')
    tree_a.commit('message 1', rev_id=b'rev1')
    tree_a.commit('message 2', rev_id=b'rev2a')
    tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
    uncommit.uncommit(tree_b.branch, tree=tree_b)
    tree_b.commit('message 2', rev_id=b'rev2b')
    self.do_full_upload(directory=tree_a.basedir)
    return tree_b