import os
import subprocess
import sys
from io import BytesIO
from dulwich.repo import Repo
from ...tests import TestCaseWithTransport
from ...tests.features import PathFeature
from ..git_remote_helper import RemoteHelper, fetch, open_local_dir
from ..object_store import get_object_store
from . import FastimportFeature
def test_from_env_dir(self):
    self.make_branch_and_tree('bla', format='git')
    self.overrideEnv('GIT_DIR', os.path.join(self.test_dir, 'bla', '.git'))
    open_local_dir()