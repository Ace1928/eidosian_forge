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
def test_permission_denied_github(self):
    e = parse_git_error('url', 'Permission to porridge/gaduhistory.git denied to jelmer.')
    self.assertIsInstance(e, PermissionDenied)
    self.assertEqual(e.path, 'porridge/gaduhistory.git')
    self.assertEqual(e.extra, ': denied to jelmer')