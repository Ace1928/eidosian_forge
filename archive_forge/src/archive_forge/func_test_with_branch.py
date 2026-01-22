from ...tests import TestCase
from ..urls import git_url_to_bzr_url
def test_with_branch(self):
    self.assertEqual(git_url_to_bzr_url('foo:bar/path', branch=''), 'git+ssh://foo/bar/path')
    self.assertEqual(git_url_to_bzr_url('foo:bar/path', branch='foo/blah'), 'git+ssh://foo/bar/path,branch=foo%2Fblah')
    self.assertEqual(git_url_to_bzr_url('foo:bar/path', branch='blah'), 'git+ssh://foo/bar/path,branch=blah')