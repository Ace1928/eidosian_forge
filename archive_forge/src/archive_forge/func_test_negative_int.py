import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_negative_int(self):
    self.assertInHistoryIs(2, b'r2', '-1')
    self.assertInHistoryIs(1, b'r1', '-2')
    self.assertInHistoryIs(1, b'r1', '-3')
    self.assertInHistoryIs(1, b'r1', '-4')
    self.assertInHistoryIs(1, b'r1', '-100')