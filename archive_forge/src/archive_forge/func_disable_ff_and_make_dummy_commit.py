import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
def disable_ff_and_make_dummy_commit(self):
    dest = repo.Repo(os.path.join(self.gitroot, 'dest'))
    run_git_or_fail(['config', 'receive.denyNonFastForwards', 'true'], cwd=dest.path)
    commit_id = self.make_dummy_commit(dest)
    return (dest, commit_id)