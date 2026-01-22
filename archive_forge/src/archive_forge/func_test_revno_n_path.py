import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_revno_n_path(self):
    """Old revno:N:path tests"""
    wta = self.make_branch_and_tree('a')
    ba = wta.branch
    wta.commit('Commit one', rev_id=b'a@r-0-1')
    wta.commit('Commit two', rev_id=b'a@r-0-2')
    wta.commit('Commit three', rev_id=b'a@r-0-3')
    wtb = self.make_branch_and_tree('b')
    bb = wtb.branch
    wtb.commit('Commit one', rev_id=b'b@r-0-1')
    wtb.commit('Commit two', rev_id=b'b@r-0-2')
    wtb.commit('Commit three', rev_id=b'b@r-0-3')
    self.assertEqual((1, b'a@r-0-1'), spec_in_history('revno:1:a/', ba))
    self.assertEqual((1, b'a@r-0-1'), spec_in_history('revno:1:a/', None))
    self.assertEqual((1, b'a@r-0-1'), spec_in_history('revno:1:a/', bb))
    self.assertEqual((2, b'b@r-0-2'), spec_in_history('revno:2:b/', None))