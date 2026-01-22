from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_revision_branch_interaction(self):
    self.assertBundleContains([b'rev3', b'rev2'], ['../grandparent'])
    self.assertBundleContains([b'rev2'], ['../grandparent', '-r-2'])
    self.assertBundleContains([b'rev3', b'rev2'], ['../grandparent', '-r-2..-1'])
    md = self.get_MD(['-r-2..-1'])
    self.assertEqual(b'rev2', md.base_revision_id)
    self.assertEqual(b'rev3', md.revision_id)