from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_uses_submit(self):
    """Submit location can be used and set"""
    self.assertBundleContains([b'rev3'], [])
    self.assertBundleContains([b'rev3', b'rev2'], ['../grandparent'])
    self.assertBundleContains([b'rev3', b'rev2'], [])
    self.run_send(['../parent'])
    self.assertBundleContains([b'rev3', b'rev2'], [])
    self.run_send(['../parent', '--remember'])
    self.assertBundleContains([b'rev3'], [])
    err = self.run_send(['--remember'], rc=3)[1]
    self.assertContainsRe(err, '--remember requires a branch to be specified.')