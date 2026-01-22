import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_in_repository(self):
    """We can get any revision id in the repository"""
    self.tree2.commit('alt third', rev_id=b'alt_r3')
    self.tree.branch.fetch(self.tree2.branch, b'alt_r3')
    self.assertInHistoryIs(None, b'alt_r3', 'revid:alt_r3')