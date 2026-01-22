import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_missing_number_and_branch(self):
    self.assertInvalid('revno::', extra='\ncannot have an empty revno and no branch')