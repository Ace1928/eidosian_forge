import unittest
from traits.testing.optional_dependencies import (
import traits
def test_version_git_revision(self):
    from traits.version import git_revision
    self.assertIsInstance(git_revision, str)
    self.assertEqual(len(git_revision), 40)
    self.assertLessEqual(set(git_revision), set('0123456789abcdef'))