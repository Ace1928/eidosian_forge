from ...tests import TestCase
from ..urls import git_url_to_bzr_url
def test_just_ssh(self):
    self.assertEqual(git_url_to_bzr_url('ssh://user@foo/bar/path'), 'git+ssh://user@foo/bar/path')