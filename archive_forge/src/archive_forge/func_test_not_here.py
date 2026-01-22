import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_not_here(self):
    self.tree2.commit('alt third', rev_id=b'alt_r3')
    self.assertInvalid('revid:alt_r3', invalid_as_revision_id=False)