import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_append_dwim_revspec(self):
    original_dwim_revspecs = list(RevisionSpec_dwim._possible_revspecs)

    def reset_dwim_revspecs():
        RevisionSpec_dwim._possible_revspecs = original_dwim_revspecs
    self.addCleanup(reset_dwim_revspecs)
    RevisionSpec_dwim.append_possible_revspec(RevisionSpec_bork)
    self.assertAsRevisionId(b'r1', 'bork')