import gzip
import re
from dulwich.refs import SymrefLoop
from .. import config, debug, errors, osutils, trace, ui, urlutils
from ..controldir import BranchReferenceLoop
from ..errors import (AlreadyBranchError, BzrError, ConnectionReset,
from ..push import PushResult
from ..revision import NULL_REVISION
from ..revisiontree import RevisionTree
from ..transport import (NoSuchFile, Transport,
from . import is_github_url, lazy_check_versions, user_agent_for_github
import os
import select
import urllib.parse as urlparse
import dulwich
import dulwich.client
from dulwich.errors import GitProtocolError, HangupException
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, Pack, load_pack_index,
from dulwich.protocol import ZERO_SHA
from dulwich.refs import SYMREF, DictRefsContainer
from dulwich.repo import NotGitRepository
from .branch import (GitBranch, GitBranchFormat, GitBranchPushResult, GitTags,
from .dir import GitControlDirFormat, GitDir
from .errors import GitSmartRemoteNotSupported
from .mapping import encode_git_path, mapping_registry
from .object_store import get_object_store
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_peeled, ref_to_tag_name,
from .repository import GitRepository, GitRepositoryFormat
def parse_git_hangup(url, e):
    """Parse the error lines from a git servers stderr on hangup.

    :param url: URL of the remote repository
    :param e: A HangupException
    """
    stderr_lines = getattr(e, 'stderr_lines', None)
    if not stderr_lines:
        return ConnectionReset('Connection closed early', e)
    if all((line.startswith(b'remote: ') for line in stderr_lines)):
        stderr_lines = [line[len(b'remote: '):] for line in stderr_lines]
    interesting_lines = [line for line in stderr_lines if line and line.replace(b'=', b'')]
    if len(interesting_lines) == 1:
        interesting_line = interesting_lines[0]
        return parse_git_error(url, interesting_line.decode('utf-8', 'surrogateescape'))
    return RemoteGitError(b'\n'.join(stderr_lines).decode('utf-8', 'surrogateescape'))