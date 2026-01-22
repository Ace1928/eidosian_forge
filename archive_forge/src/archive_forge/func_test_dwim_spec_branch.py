import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_dwim_spec_branch(self):
    self.assertInHistoryIs(None, b'alt_r2', 'tree2')