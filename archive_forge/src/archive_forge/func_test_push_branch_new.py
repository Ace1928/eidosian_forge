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
def test_push_branch_new(self):
    remote = ControlDir.open(self.remote_url)
    wt = self.make_branch_and_tree('local', format=self._from_format)
    self.build_tree(['local/blah'])
    wt.add(['blah'])
    revid = wt.commit('blah')
    if self._from_format == 'git':
        result = remote.push_branch(wt.branch, name='newbranch')
    else:
        result = remote.push_branch(wt.branch, lossy=True, name='newbranch')
    self.assertEqual(0, result.old_revno)
    if self._from_format == 'git':
        self.assertEqual(1, result.new_revno)
    else:
        self.assertIs(None, result.new_revno)
    result.report(BytesIO())
    self.assertEqual({b'refs/heads/newbranch': self.remote_real.refs[b'refs/heads/newbranch']}, self.remote_real.get_refs())