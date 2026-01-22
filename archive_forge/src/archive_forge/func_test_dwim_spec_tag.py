import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_dwim_spec_tag(self):
    self.tree.branch.tags.set_tag('footag', b'r1')
    self.assertAsRevisionId(b'r1', 'footag')
    self.tree.branch.tags.delete_tag('footag')
    self.assertRaises(InvalidRevisionSpec, self.get_in_history, 'footag')