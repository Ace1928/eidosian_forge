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
def test_http_unexpected(self):
    self.assertEqual(UnexpectedHttpStatus('https://example.com/bigint.git/git-upload-pack', 403, extra='unexpected http resp 403 for https://example.com/bigint.git/git-upload-pack'), parse_git_error('url', RemoteGitError('unexpected http resp 403 for https://example.com/bigint.git/git-upload-pack')))