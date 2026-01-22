import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_from_string_tag(self):
    spec = RevisionSpec.from_string('tag:bzr-0.14')
    self.assertIsInstance(spec, RevisionSpec_tag)
    self.assertEqual(spec.spec, 'bzr-0.14')