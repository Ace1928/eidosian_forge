import gzip
import os
import time
from io import BytesIO
from dulwich import porcelain
from dulwich.errors import HangupException
from dulwich.repo import Repo as GitRepo
from ...branch import Branch
from ...controldir import BranchReferenceLoop, ControlDir
from ...errors import (ConnectionReset, DivergedBranches, NoSuchTag,
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ExecutableFeature
from ...urlutils import join as urljoin
from ..mapping import default_mapping
from ..remote import (GitRemoteRevisionTree, GitSmartRemoteNotSupported,
from ..tree import MissingNestedTree
def test_push_branch_symref(self):
    cfg = self.remote_real.get_config()
    cfg.set((b'core',), b'bare', True)
    cfg.write_to_path()
    self.remote_real.refs.set_symbolic_ref(b'HEAD', b'refs/heads/master')
    c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/heads/master')
    remote = ControlDir.open(self.remote_url)
    wt = self.make_branch_and_tree('local', format=self._from_format)
    self.build_tree(['local/blah'])
    wt.add(['blah'])
    revid = wt.commit('blah')
    if self._from_format == 'git':
        result = remote.push_branch(wt.branch, overwrite=True)
    else:
        result = remote.push_branch(wt.branch, lossy=True, overwrite=True)
    self.assertEqual(None, result.old_revno)
    if self._from_format == 'git':
        self.assertEqual(1, result.new_revno)
    else:
        self.assertIs(None, result.new_revno)
    result.report(BytesIO())
    self.assertEqual({b'HEAD': self.remote_real.refs[b'refs/heads/master'], b'refs/heads/master': self.remote_real.refs[b'refs/heads/master']}, self.remote_real.get_refs())