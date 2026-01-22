import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_invalid_revno_in_branch(self):
    self.tree.commit('three', rev_id=b'r3')
    self.assertInvalid('revno:3:tree2')