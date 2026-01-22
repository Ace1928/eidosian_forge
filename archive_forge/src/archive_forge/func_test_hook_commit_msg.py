import os
import shutil
import stat
import sys
import tempfile
from dulwich import errors
from dulwich.tests import TestCase
from ..hooks import CommitMsgShellHook, PostCommitShellHook, PreCommitShellHook
def test_hook_commit_msg(self):
    repo_dir = os.path.join(tempfile.mkdtemp())
    os.mkdir(os.path.join(repo_dir, 'hooks'))
    self.addCleanup(shutil.rmtree, repo_dir)
    commit_msg_fail = '#!/bin/sh\nexit 1\n'
    commit_msg_success = '#!/bin/sh\nexit 0\n'
    commit_msg_cwd = '#!/bin/sh\nif [ "$(pwd)" = \'' + repo_dir + "' ]; then exit 0; else exit 1; fi\n"
    commit_msg = os.path.join(repo_dir, 'hooks', 'commit-msg')
    hook = CommitMsgShellHook(repo_dir)
    with open(commit_msg, 'w') as f:
        f.write(commit_msg_fail)
    os.chmod(commit_msg, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    self.assertRaises(errors.HookError, hook.execute, b'failed commit')
    if sys.platform != 'darwin':
        with open(commit_msg, 'w') as f:
            f.write(commit_msg_cwd)
        os.chmod(commit_msg, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        hook.execute(b'cwd test commit')
    with open(commit_msg, 'w') as f:
        f.write(commit_msg_success)
    os.chmod(commit_msg, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    hook.execute(b'empty commit')