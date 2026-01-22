import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_in_history(self):
    self.assertInHistoryIs(2, b'r2', 'mainline:revid:alt_r2')