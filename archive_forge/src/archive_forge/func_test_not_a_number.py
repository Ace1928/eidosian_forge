import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_not_a_number(self):
    last_e = None
    try:
        int('Y')
    except ValueError as e:
        last_e = e
    self.assertInvalid('last:Y', extra='\n' + str(last_e))