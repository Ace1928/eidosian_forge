import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_wants_no_revision_history(self):
    spec = RevisionSpecMatchOnTrap('foo', _internal=True)
    spec.in_history(self.tree.branch)
    self.assertEqual((self.tree.branch, None), spec.last_call)