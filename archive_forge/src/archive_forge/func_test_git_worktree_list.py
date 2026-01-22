import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
def test_git_worktree_list(self):
    require_git_version((2, 7, 0))
    output = run_git_or_fail(['worktree', 'list'], cwd=self._repo.path)
    worktrees = self._parse_worktree_list(output)
    self.assertEqual(len(worktrees), self._number_of_working_tree)
    self.assertEqual(worktrees[0][1], '(bare)')
    self.assertTrue(os.path.samefile(worktrees[0][0], self._mainworktree_repo.path))
    output = run_git_or_fail(['worktree', 'list'], cwd=self._mainworktree_repo.path)
    worktrees = self._parse_worktree_list(output)
    self.assertEqual(len(worktrees), self._number_of_working_tree)
    self.assertEqual(worktrees[0][1], '(bare)')
    self.assertTrue(os.path.samefile(worktrees[0][0], self._mainworktree_repo.path))