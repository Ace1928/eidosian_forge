from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_content_options(self):
    """--no-patch and --no-bundle should work and be independant"""
    md = self.get_MD([])
    self.assertIsNot(None, md.bundle)
    self.assertIsNot(None, md.patch)
    md = self.get_MD(['--format=0.9'])
    self.assertIsNot(None, md.bundle)
    self.assertIsNot(None, md.patch)
    md = self.get_MD(['--no-patch'])
    self.assertIsNot(None, md.bundle)
    self.assertIs(None, md.patch)
    self.run_bzr_error(['Format 0.9 does not permit bundle with no patch'], ['send', '--no-patch', '--format=0.9', '-o-'], working_dir='branch')
    md = self.get_MD(['--no-bundle', '.', '.'])
    self.assertIs(None, md.bundle)
    self.assertIsNot(None, md.patch)
    md = self.get_MD(['--no-bundle', '--format=0.9', '../parent', '.'])
    self.assertIs(None, md.bundle)
    self.assertIsNot(None, md.patch)
    md = self.get_MD(['--no-bundle', '--no-patch', '.', '.'])
    self.assertIs(None, md.bundle)
    self.assertIs(None, md.patch)
    md = self.get_MD(['--no-bundle', '--no-patch', '--format=0.9', '../parent', '.'])
    self.assertIs(None, md.bundle)
    self.assertIs(None, md.patch)